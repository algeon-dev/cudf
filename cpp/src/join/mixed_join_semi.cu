/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_filter_join_common_utils.cuh"
#include "mixed_join_kernels_semi.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <cuda/atomic>

#include <memory>
#include <optional>

namespace cudf {
namespace detail {

namespace {
template <typename Hasher>
CUDF_KERNEL void mixed_left_build(Hasher hash,
                                  size_type rows,
                                  size_type* heads,
                                  size_type* next,
                                  std::size_t mask)
{
  auto const row = cudf::detail::grid_1d::global_thread_id<256>();
  if (row >= rows) { return; }
  auto& head = heads[hash(row) & mask];
  next[row] = cuda::atomic_ref<size_type, cuda::thread_scope_device>{head}.exchange(
    row, cuda::memory_order_relaxed);
}

template <typename Hasher, typename Equal>
CUDF_KERNEL void mixed_left_consume(Hasher hash,
                                    Equal equal,
                                    table_device_view left,
                                    table_device_view right,
                                    size_type const* heads,
                                    size_type const* next,
                                    std::size_t mask,
                                    uint32_t* matches,
                                    ast::detail::expression_device_view expression)
{
  extern __shared__ char raw_intermediates[];
  auto* intermediates = reinterpret_cast<ast::detail::IntermediateDataType<true>*>(raw_intermediates) +
                        threadIdx.x * expression.num_intermediates;
  auto const evaluator = ast::detail::expression_evaluator<true>(left, right, expression);
  auto const right_row = cudf::detail::grid_1d::global_thread_id<128>();
  if (right_row >= right.num_rows()) { return; }
  for (auto left_row = heads[hash(right_row) & mask]; left_row != JoinNoMatch;
       left_row = next[left_row]) {
    if (!equal(row::lhs_index_type{left_row}, row::rhs_index_type{right_row})) { continue; }
    auto matched = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{matches[left_row]};
    if (matched.load(cuda::memory_order_relaxed) != 0) { continue; }
    auto result = ast::detail::value_expression_result<bool, true>();
    evaluator.evaluate(result, left_row, right_row, 0, intermediates);
    if (result.is_valid() && result.value()) { matched.store(1, cuda::memory_order_relaxed); }
  }
}

std::size_t mixed_left_bucket_count(size_type rows)
{
  std::size_t count = 1;
  while (count < static_cast<std::size_t>(rows)) { count *= 2; }
  return count;
}
}  // namespace

class mixed_left_join_state {
 public:
  mixed_left_join_state(table_view const& left_equality,
                        table_view const& left_conditional,
                        null_equality compare_nulls,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
    : _left{left_equality},
      _conditional{left_conditional},
      _nulls{compare_nulls},
      _preprocessed{row::equality::preprocessed_table::create(left_equality, stream, mr)},
      _conditional_device{table_device_view::create(left_conditional, stream, mr)},
      _heads{mixed_left_bucket_count(left_equality.num_rows()), stream, mr},
      _next{static_cast<std::size_t>(left_equality.num_rows()), stream, mr},
      _matches{static_cast<std::size_t>(left_equality.num_rows()), stream, mr}
  {
    CUDF_EXPECTS(_left.num_columns() > 0, "mixed left state requires equality keys");
    CUDF_EXPECTS(_left.num_rows() == _conditional.num_rows(), "left table row counts differ");
    CUDF_EXPECTS(!cudf::has_nested_columns(_left), "nested mixed equality keys are unsupported");
    thrust::fill(rmm::exec_policy_nosync(stream, mr), _heads.begin(), _heads.end(), JoinNoMatch);
    thrust::fill(rmm::exec_policy_nosync(stream, mr), _matches.begin(), _matches.end(), uint32_t{0});
    if (_left.num_rows() == 0) { return; }
    auto hash = row::hash::row_hasher{_preprocessed}.device_hasher(nullate::YES{});
    auto config = grid_1d{_left.num_rows(), 256};
    mixed_left_build<<<config.num_blocks, 256, 0, stream.value()>>>(
      hash, _left.num_rows(), _heads.data(), _next.data(), _heads.size() - 1);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  void consume(table_view const& right_equality,
               table_view const& right_conditional,
               ast::expression const& predicate,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
  {
    CUDF_EXPECTS(!_sealed, "mixed left state is already sealed");
    CUDF_EXPECTS(right_equality.num_rows() == right_conditional.num_rows(), "right row counts differ");
    CUDF_EXPECTS(have_same_types(_left, right_equality), "mixed equality key types differ");
    auto parser = ast::detail::expression_parser{
      predicate, _conditional, right_conditional, nullate::YES{}, stream, mr};
    CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8, "mixed predicate must return boolean");
    if (_left.num_rows() == 0 || right_equality.num_rows() == 0) { return; }
    auto right = row::equality::preprocessed_table::create(right_equality, stream, mr);
    auto conditional = table_device_view::create(right_conditional, stream, mr);
    auto hash = row::hash::row_hasher{right}.device_hasher(nullate::YES{});
    auto equal = row::equality::two_table_comparator{_preprocessed, right}.equal_to<false>(
      nullate::YES{}, _nulls);
    auto config = grid_1d{right_equality.num_rows(), 128};
    auto const shared_bytes = 128 * parser.shmem_per_thread;
    mixed_left_consume<<<config.num_blocks, 128, shared_bytes, stream.value()>>>(
      hash, equal, *_conditional_device, *conditional, _heads.data(), _next.data(),
      _heads.size() - 1, _matches.data(), parser.device_expression_data);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  void seal(bool anti, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    CUDF_EXPECTS(!_sealed, "mixed left state is already sealed");
    _output_size = thrust::count(rmm::exec_policy(stream, mr), _matches.begin(), _matches.end(),
                                 anti ? uint32_t{0} : uint32_t{1});
    _anti = anti;
    _sealed = true;
  }

  std::size_t output_size() const
  {
    CUDF_EXPECTS(_sealed, "mixed left state must be sealed before output");
    return _output_size;
  }

  std::unique_ptr<rmm::device_uvector<size_type>> materialize_indices(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    auto output = std::make_unique<rmm::device_uvector<size_type>>(output_size(), stream, mr);
    auto matches = _matches.data();
    auto wanted = _anti ? uint32_t{0} : uint32_t{1};
    thrust::copy_if(rmm::exec_policy_nosync(stream, mr),
                    thrust::counting_iterator<size_type>{0},
                    thrust::counting_iterator<size_type>{_left.num_rows()}, output->begin(),
                    [matches, wanted] __device__(size_type row) { return matches[row] == wanted; });
    return output;
  }

 private:
  table_view _left;
  table_view _conditional;
  null_equality _nulls;
  std::shared_ptr<row::equality::preprocessed_table> _preprocessed;
  std::unique_ptr<table_device_view, std::function<void(table_device_view*)>> _conditional_device;
  rmm::device_uvector<size_type> _heads;
  rmm::device_uvector<size_type> _next;
  rmm::device_uvector<uint32_t> _matches;
  bool _sealed{false};
  bool _anti{false};
  std::size_t _output_size{0};
};

namespace {
struct mixed_single_gather_mask {
  join_kind kind;
  device_span<bool const> keep_mask;

  __device__ bool operator()(size_type row_index) const noexcept
  {
    return keep_mask[row_index] == (kind == join_kind::LEFT_SEMI_JOIN);
  }
};
}  // namespace

class mixed_left_semi_anti_join_size_data {
 public:
  explicit mixed_left_semi_anti_join_size_data(size_type output_size) : _output_size{output_size} {}

  mixed_left_semi_anti_join_size_data(join_kind kind,
                                      size_type left_rows,
                                      size_type output_size,
                                      rmm::device_uvector<bool> keep_mask)
    : _kind{kind},
      _left_rows{left_rows},
      _output_size{output_size},
      _keep_mask{std::move(keep_mask)}
  {
  }

  [[nodiscard]] std::size_t output_size() const { return static_cast<std::size_t>(_output_size); }

  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> materialize_indices(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    auto result = std::make_unique<rmm::device_uvector<size_type>>(
      static_cast<std::size_t>(_output_size), stream, mr);
    if (!_keep_mask.has_value()) {
      thrust::sequence(rmm::exec_policy_nosync(stream, mr), result->begin(), result->end());
      return result;
    }

    thrust::copy_if(
      rmm::exec_policy_nosync(stream, mr),
      thrust::counting_iterator<size_type>(0),
      thrust::counting_iterator<size_type>(_left_rows),
      result->begin(),
      mixed_single_gather_mask{
        _kind, device_span<bool const>(_keep_mask->data(), static_cast<std::size_t>(_left_rows))});
    return result;
  }

 private:
  join_kind _kind = join_kind::LEFT_SEMI_JOIN;
  size_type _left_rows{};
  size_type _output_size{};
  std::optional<rmm::device_uvector<bool>> _keep_mask{};
};

std::unique_ptr<mixed_left_semi_anti_join_size_data> compute_mixed_join_semi_size_data(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((join_type != join_kind::INNER_JOIN) and (join_type != join_kind::LEFT_JOIN) and
                 (join_type != join_kind::FULL_JOIN),
               "Inner, left, and full joins should use mixed_join.");

  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  auto const right_num_rows{right_conditional.num_rows()};
  auto const left_num_rows{left_conditional.num_rows()};
  auto const outer_num_rows{left_num_rows};

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  // The retained empty states intentionally hold no device mask.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Anti returns all the row indices from left; no matches can exist.
      case join_kind::LEFT_ANTI_JOIN:
        return std::make_unique<mixed_left_semi_anti_join_size_data>(left_num_rows);
      // Semi returns empty output because no matches can exist.
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<mixed_left_semi_anti_join_size_data>(0);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Anti and semi joins both return empty sets.
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<mixed_left_semi_anti_join_size_data>(0);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) or cudf::has_nulls(right_equality) or
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& left                  = left_equality;
  auto& right                 = right_equality;
  auto left_view              = table_device_view::create(left, stream, mr);
  auto right_view             = table_device_view::create(right, stream, mr);
  auto left_conditional_view  = table_device_view::create(left_conditional, stream, mr);
  auto right_conditional_view = table_device_view::create(right_conditional, stream, mr);

  auto const preprocessed_right =
    cudf::detail::row::equality::preprocessed_table::create(right, stream, mr);
  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream, mr);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_left, preprocessed_right};
  auto const equality_left = row_comparator.equal_to<false>(has_nulls, compare_nulls);

  // Create hash table containing all keys found in right table
  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const right_nulls    = cudf::nullate::DYNAMIC{cudf::has_nulls(right)};
  auto const row_hash_right = cudf::detail::row::hash::row_hasher{preprocessed_right};

  // Since we may see multiple rows that are identical in the equality tables
  // but differ in the conditional tables, the equality comparator used for
  // insertion must account for both sets of tables. An alternative solution
  // would be to use a multimap, but that solution would store duplicates where
  // equality and conditional rows are equal, so this approach is preferable.
  // One way to make this solution even more efficient would be to only include
  // the columns of the conditional table that are used by the expression, but
  // that requires additional plumbing through the AST machinery and is out of
  // scope for now.
  auto const row_comparator_right =
    cudf::detail::row::equality::two_table_comparator{preprocessed_right, preprocessed_right};
  auto const equality_right_equality =
    row_comparator_right.equal_to<false>(right_nulls, compare_nulls);
  auto const preprocessed_right_condtional =
    cudf::detail::row::equality::preprocessed_table::create(right_conditional, stream, mr);
  auto const row_comparator_conditional_right = cudf::detail::row::equality::two_table_comparator{
    preprocessed_right_condtional, preprocessed_right_condtional};
  auto const equality_right_conditional =
    row_comparator_conditional_right.equal_to<false>(right_nulls, compare_nulls);

  hash_set_type row_set{{static_cast<std::size_t>(right.num_rows())},
                        cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                        cuco::empty_key{JoinNoMatch},
                        {equality_right_equality, equality_right_conditional},
                        {row_hash_right.device_hasher(right_nulls)},
                        {},
                        {},
                        rmm::mr::polymorphic_allocator<char>{mr},
                        {stream.value()}};

  auto iter = cuda::counting_iterator<cudf::size_type>{0};

  // skip rows that are null here.
  if ((compare_nulls == null_equality::EQUAL) or (not nullable(right))) {
    row_set.insert_async(iter, iter + right_num_rows, stream.value());
  } else {
    cuda::counting_iterator<cudf::size_type> stencil(0);
    auto const [row_bitmask, _] = cudf::detail::bitmask_and(right, stream, mr);
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    row_set.insert_if_async(iter, iter + right_num_rows, stencil, pred, stream.value());
  }

  detail::grid_1d const config(outer_num_rows * hash_set_type::cg_size, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block =
    parser.shmem_per_thread *
    cuco::detail::int_div_ceil(config.num_threads_per_block, hash_set_type::cg_size);

  auto const row_hash  = cudf::detail::row::hash::row_hasher{preprocessed_left};
  auto const hash_left = row_hash.device_hasher(has_nulls);

  hash_set_ref_type const row_set_ref = row_set.ref(cuco::contains).rebind_hash_function(hash_left);

  // Vector used to indicate indices from the left table which are present in output
  auto left_table_keep_mask =
    rmm::device_uvector<bool>(static_cast<std::size_t>(left.num_rows()), stream, mr);

  launch_mixed_join_semi(has_nulls,
                         *left_conditional_view,
                         *right_conditional_view,
                         *left_view,
                         *right_view,
                         equality_left,
                         row_set_ref,
                         cudf::device_span<bool>(left_table_keep_mask),
                         parser.device_expression_data,
                         config,
                         shmem_size_per_block,
                         stream);

  auto const output_size = static_cast<size_type>(thrust::count_if(
    rmm::exec_policy_nosync(stream, mr),
    thrust::counting_iterator<size_type>(0),
    thrust::counting_iterator<size_type>(left_num_rows),
    mixed_single_gather_mask{join_type,
                             device_span<bool const>(left_table_keep_mask.data(),
                                                     static_cast<std::size_t>(left_num_rows))}));

  return std::make_unique<mixed_left_semi_anti_join_size_data>(
    join_type, left_num_rows, output_size, std::move(left_table_keep_mask));
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_join_semi(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto state = compute_mixed_join_semi_size_data(left_equality,
                                                 right_equality,
                                                 left_conditional,
                                                 right_conditional,
                                                 binary_predicate,
                                                 compare_nulls,
                                                 join_type,
                                                 stream,
                                                 mr);
  return state->materialize_indices(stream, mr);
}

}  // namespace detail

mixed_left_join_state::mixed_left_join_state(table_view const& equality,
                                             table_view const& conditional,
                                             null_equality nulls,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
  : _impl{std::make_unique<detail::mixed_left_join_state>(equality, conditional, nulls, stream, mr)}
{
}
mixed_left_join_state::~mixed_left_join_state() = default;
void mixed_left_join_state::consume(table_view const& equality,
                                    table_view const& conditional,
                                    ast::expression const& predicate,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  _impl->consume(equality, conditional, predicate, stream, mr);
}
void mixed_left_join_state::seal(bool anti, rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  _impl->seal(anti, stream, mr);
}
std::size_t mixed_left_join_state::output_size() const { return _impl->output_size(); }
std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_join_state::materialize_indices(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return _impl->materialize_indices(stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_semi_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_SEMI_JOIN,
                                 stream,
                                 mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_anti_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_ANTI_JOIN,
                                 stream,
                                 mr);
}

mixed_left_semi_anti_join_size_data::~mixed_left_semi_anti_join_size_data() = default;

mixed_left_semi_anti_join_size_data::mixed_left_semi_anti_join_size_data(
  std::unique_ptr<cudf::detail::mixed_left_semi_anti_join_size_data> impl)
  : _impl{std::move(impl)}
{
}

std::size_t mixed_left_semi_anti_join_size_data::output_size() const
{
  return _impl->output_size();
}

std::unique_ptr<rmm::device_uvector<size_type>>
mixed_left_semi_anti_join_size_data::materialize_indices(rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr) const
{
  return _impl->materialize_indices(stream, mr);
}

std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_semi_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::unique_ptr<mixed_left_semi_anti_join_size_data>{
    new mixed_left_semi_anti_join_size_data{
      detail::compute_mixed_join_semi_size_data(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::LEFT_SEMI_JOIN,
                                                stream,
                                                mr)}};
}

std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_anti_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::unique_ptr<mixed_left_semi_anti_join_size_data>{
    new mixed_left_semi_anti_join_size_data{
      detail::compute_mixed_join_semi_size_data(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::LEFT_ANTI_JOIN,
                                                stream,
                                                mr)}};
}

}  // namespace cudf
