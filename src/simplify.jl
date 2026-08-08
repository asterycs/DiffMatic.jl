# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function simplify(arg::Value)
    return arg
end

function simplify(arg::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(simplify(evaluate(arg.arg)))
end

function simplify(arg::BinaryOperation{Op}) where {Op}
    simplified =
        evaluate(simplify(Op(), simplify(evaluate(arg.arg1)), simplify(evaluate(arg.arg2))))

    # Ensure the output is consistent.
    if !issetequal(get_free_indices(simplified), get_free_indices(arg))
        @assert false "Something went wrong, please open an issue with your input."
    end

    return simplified
end

function elementwise_indices(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return intersect(arg1_indices, arg2_indices)
end

function get_diag_delta(arg::BinaryOperation{Mult})
    l = get_diag_delta(arg.arg1)
    r = get_diag_delta(arg.arg2)

    if !isnothing(l)
        return l
    elseif !isnothing(r)
        return r
    end

    return nothing
end

function get_diag_delta(arg::KrD)
    if first(arg.indices).letter != last(arg.indices).letter
        return arg
    end

    return nothing
end

function get_diag_delta(arg)
    return nothing
end

function to_binary_operation(op::Op, terms::AbstractArray) where {Op}
    if length(terms) == 1
        return first(terms)
    end

    return BinaryOperation{Op}(to_binary_operation(op, terms[1:(end-1)]), terms[end])
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::Literal)
    op = BinaryOperation{Mult}(arg1, arg2)

    op = invoke(simplify, Tuple{Mult,BinaryOperation{Mult},Tensor}, Mult(), arg1, arg2)

    arg2_free_ids = get_free_indices(arg2)

    if !can_contract(arg1, arg2) || length(arg2_free_ids) != 1
        return op
    end

    target_idx = only(arg2_free_ids)
    tied_idx = flip(target_idx)
    factors = collect_factors(arg1)

    carriers = [k for k ∈ eachindex(factors) if tied_idx ∈ get_free_indices(factors[k])]

    if length(carriers) < 2
        return op
    end

    # Find element-wise product of matrices
    rest = [i for k ∈ carriers for i ∈ get_free_indices(factors[k]) if i != tied_idx]

    if length(unique(rest)) != length(rest)
        return op
    end

    delta = findfirst(k -> factors[k] isa KrD, carriers)
    reshaped = nothing

    if isnothing(delta)
        # E.g. x¹y¹1₁
        chosen = first(carriers)
        factors[chosen] =
            update_index(factors[chosen], tied_idx, target_idx; allow_shape_change = true)
        reshaped = evaluate(to_binary_operation(Mult(), factors))
    else
        # E.g. arg1 = x₁δ₁² arg2 = 1¹
        k = carriers[delta]
        wire_ids = filter(!isequal(tied_idx), get_free_indices(factors[k]))

        if length(wire_ids) != 1
            return op
        end

        wire = only(wire_ids)
        merged = Any[]

        for j ∈ eachindex(factors)
            if j == k
                continue
            end

            factor = factors[j]

            if j ∈ carriers
                factor = update_index(factors[j], tied_idx, wire; allow_shape_change = true)
            end

            push!(merged, factor)
        end

        reshaped = evaluate(to_binary_operation(Mult(), merged))
    end

    if !issetequal(get_free_indices(reshaped), get_free_indices(op))
        return op
    end

    if arg2.value != 1
        reshaped = BinaryOperation{Mult}(arg2.value, reshaped)
    end

    return simplify(reshaped)
end

function simplify(::Mult, arg1::Tensor, arg2::BinaryOperation{Mult})
    return simplify(Mult(), arg2, arg1)
end

function can_apply(l::BinaryOperation{Mult}, r::KrD)
    return can_contract(l.arg1, r) || can_contract(l.arg2, r)
end

function can_apply(l::KrD, r::BinaryOperation{Mult})
    return can_contract(l, r.arg1) || can_contract(l, r.arg2)
end

function can_apply(l::BinaryOperation{Mult}, r::BinaryOperation{Mult})
    return can_apply(l, r.arg1) || can_apply(l, r.arg2)
end

function can_apply(l::BinaryOperation{Mult}, r::Tensor)
    return can_contract(l.arg1, r) || can_contract(l.arg2, r)
end

function can_apply(l::Tensor, r::BinaryOperation{Mult})
    return can_contract(l, r.arg1) || can_contract(l, r.arg2)
end

function can_apply(l::Tensor, r::KrD)
    return can_contract(l, r)
end

function can_apply(l::KrD, r::Tensor)
    return can_contract(l, r)
end

function can_apply(l::KrD, r::KrD)
    return can_contract(l, r)
end

function can_apply(l, r)
    return false
end

function build_graph(factors)
    pendants = Dict{Int,Vector{Any}}()
    edges = Vector{Tuple{Any,Vector{Int}}}()
    scalars = Any[]

    for f ∈ factors
        ids = get_free_indices(f)

        if isempty(ids)
            push!(scalars, f)
            continue
        end

        if length(ids) > 2
            return nothing
        end

        letters = [i.letter for i ∈ ids]

        # A factor carrying the same letter twice is a trace.
        if length(letters) != length(unique(letters))
            return nothing
        end

        if length(ids) == 1
            push!(get!(pendants, only(letters), Any[]), f)
        else
            push!(edges, (f, letters))
        end
    end

    return (pendants, edges, scalars)
end

function set_letter_variance(f, letter, upper)
    for i ∈ get_free_indices(f)
        if i.letter != letter
            continue
        end

        target = upper ? Upper(letter) : Lower(letter)

        if i != target
            f = update_index(f, i, target; allow_shape_change = true)
        end
    end

    return f
end

function node_degrees(edges)
    degree = Dict{Int,Int}()

    for (_, ls) ∈ edges, l ∈ ls
        degree[l] = get(degree, l, 0) + 1
    end

    return degree
end

function find_dead_end(pendants, edges, pinned)
    degree = node_degrees(edges)

    for (k, (_, ls)) ∈ enumerate(edges)
        for (near, far) ∈ ((ls[1], ls[2]), (ls[2], ls[1]))
            if degree[near] == 1 && !(near ∈ pinned)
                return (k, near, far)
            end
        end
    end

    return nothing
end

function collapse_dead_ends!(pendants, edges, pinned)
    while true
        dead = find_dead_end(pendants, edges, pinned)

        if isnothing(dead)
            return
        end

        k, near, far = dead
        f, _ = edges[k]

        parts = Any[set_letter_variance(p, near, true) for p ∈ get(pendants, near, Any[])]
        push!(parts, set_letter_variance(f, near, false))
        folded = to_binary_operation(Mult(), parts)
        delete!(pendants, near)
        deleteat!(edges, k)
        push!(get!(pendants, far, Any[]), folded)
    end
end

function walk_path(pendants, edges, start)
    visited_edges = Set{Int}()
    nodes = Int[]
    path_edges = Any[]
    node = start

    while true
        if node ∈ nodes
            return nothing
        end

        push!(nodes, node)
        next = nothing

        for (j, (_, js)) ∈ enumerate(edges)
            if j ∉ visited_edges && node ∈ js
                next = j
                break
            end
        end

        if isnothing(next)
            break
        end

        g, gs = edges[next]
        push!(visited_edges, next)
        push!(path_edges, g)
        node = only(filter(!isequal(node), gs))
    end

    if length(visited_edges) != length(edges)
        return nothing
    end

    return (nodes, path_edges)
end

function letter_variance(f, letter)
    ids = filter(i -> i.letter == letter, get_free_indices(f))

    return isempty(ids) ? nothing : first(ids)
end

function orient(nodes, path_edges, pendants, pinned)
    if isempty(path_edges)
        ps = get(pendants, only(nodes), Any[])

        return isempty(ps) ? nothing : Vector{Any}[ps]
    end

    anchor = letter_variance(first(path_edges), first(nodes))

    if isnothing(anchor)
        return nothing
    end

    tie_upper = anchor isa Lower
    last_node = last(nodes)

    if last_node ∈ pinned
        far = letter_variance(last(path_edges), last_node)

        if isnothing(far) || (far isa Upper) != tie_upper
            return nothing
        end
    end

    groups = Vector{Any}[]

    for (k, n) ∈ enumerate(nodes)
        terminal = k == length(nodes)
        ps = get(pendants, n, Any[])

        if !isempty(ps)
            if n ∈ pinned
                push!(groups, Any[p for p ∈ ps])
            else
                v = terminal ? !tie_upper : tie_upper
                push!(groups, Any[set_letter_variance(p, n, v) for p ∈ ps])
            end
        end

        if terminal
            continue
        end

        e = path_edges[k]
        m = nodes[k+1]

        if !(n ∈ pinned)
            e = set_letter_variance(e, n, !tie_upper)
        end

        if !(m ∈ pinned)
            e = set_letter_variance(e, m, tie_upper)
        end

        push!(groups, Any[e])
    end

    return groups
end

# Factor graph (https://www.eigentales.com/Factor-Graphs/) based reordering.
function graph_rewrite(arg1, arg2, target_indices)
    factors = [collect_factors(arg1); collect_factors(arg2)]
    pinned = Set(i.letter for i ∈ target_indices)

    built = build_graph(factors)

    if isnothing(built)
        return nothing
    end

    pendants, edges, scalars = built

    collapse_dead_ends!(pendants, edges, pinned)

    degree = node_degrees(edges)

    if any(d -> d > 2, values(degree))
        return nothing
    end

    nodes = union(Set(keys(pendants)), Set(l for (_, ls) ∈ edges for l ∈ ls))

    starts = if isempty(edges)
        length(nodes) == 1 ? collect(nodes) : Int[]
    else
        [l for l ∈ pinned if get(degree, l, 0) == 1]
    end

    for start ∈ starts
        walked = walk_path(pendants, edges, start)

        if isnothing(walked)
            continue
        end

        visited, path_edges = walked

        if !issubset(nodes, Set(visited))
            continue
        end

        groups = orient(visited, path_edges, pendants, pinned)

        if isnothing(groups)
            continue
        end

        emitted = [to_binary_operation(Mult(), g) for g ∈ groups]
        rewritten = to_binary_operation(Mult(), [scalars; emitted])

        if issetequal(get_free_indices(rewritten), target_indices)
            return rewritten
        end
    end

    return nothing
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::Tensor)
    op = BinaryOperation{Mult}(arg1, arg2)
    target_indices = unique(get_free_indices(op))

    rewritten = graph_rewrite(arg1, arg2, target_indices)

    if !isnothing(rewritten)
        return rewritten
    end

    if is_diagm(arg1) &&
       !is_elementwise_multiplication(arg1, arg2) &&
       length(get_free_indices(op)) == 1
        d = get_diag_delta(arg1)

        @assert !isnothing(d)

        factors = collect_factors(arg1)
        vector_factors = filter(f -> f != d, factors)
        reshaped = []

        for f ∈ factors
            if isequal(f, d)
                continue
            end

            free_ids = get_free_indices(f)

            if isempty(free_ids)
                push!(reshaped, f)
            elseif length(free_ids) == 1 || length(free_ids) == 2
                @assert length(target_indices) == 1

                vector_index =
                    only(get_free_indices(to_binary_operation(Mult(), vector_factors)))

                current_idx = intersect(free_ids, [vector_index])

                if !isempty(current_idx)
                    f = update_index(
                        f,
                        vector_index,
                        only(target_indices);
                        allow_shape_change = true,
                    )
                end

                push!(reshaped, f)
            else
                @assert false "Not implemented, please open an issue with your input"
            end
        end

        arg2_ids = get_free_indices(arg2)

        if length(arg2_ids) == 1
            arg2 = update_index(
                arg2,
                only(arg2_ids),
                only(target_indices);
                allow_shape_change = true,
            )
            push!(reshaped, arg2)
        else
            @assert false "Not implemented, please open an issue with your input"
        end

        return to_binary_operation(Mult(), reshaped)
    end

    if length(get_free_indices(arg1)) > 2 && length(get_free_indices(arg2)) == 1
        if can_apply(arg1.arg1, arg2) &&
           can_apply(arg1.arg2, arg2) &&
           arg1.arg1 isa KrD &&
           arg1.arg2 isa KrD
            r_free_indices = get_free_indices(arg2)

            new_r = update_index(
                arg2,
                only(r_free_indices),
                first(target_indices);
                allow_shape_change = true,
            )

            eliminated = eliminated_indices([get_free_indices(arg1); r_free_indices])
            new_l = update_index(
                arg1.arg2,
                first(eliminated),
                first(target_indices);
                allow_shape_change = true,
            )

            return BinaryOperation{Mult}(new_l, new_r)
        end
    end

    return op
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    if is_diagm(arg1)
        return invoke(
            simplify,
            Tuple{Mult,BinaryOperation{Mult},Tensor},
            Mult(),
            arg1,
            arg2,
        )
    elseif is_diagm(arg2)
        return invoke(
            simplify,
            Tuple{Mult,BinaryOperation{Mult},Tensor},
            Mult(),
            arg2,
            arg1,
        )
    end

    return invoke(simplify, Tuple{Mult,BinaryOperation{Mult},Tensor}, Mult(), arg1, arg2)
end

function simplify(::Mult, arg1::KrD, arg2::KrD)
    if !can_contract(arg1, arg2)
        return BinaryOperation{Mult}(arg1, arg2)
    end

    common = indices_in_common(arg1, arg2)

    if isempty(common)
        return _multiply_with_krd(arg1, arg2)
    end

    if length(common) != 1
        return BinaryOperation{Mult}(arg1, arg2)
    end

    return Literal(1, only(common))
end

function simplify(::Mult, arg1::Value, arg2::Value)
    return evaluate(
        BinaryOperation{Mult}(evaluate(simplify(arg1)), evaluate(simplify(arg2))),
    )
end

function simplify(::Op, arg1::Value, arg2::Value) where {Op<:AdditiveOperation}
    return evaluate(BinaryOperation{Op}(simplify(evaluate(arg1)), simplify(evaluate(arg2))))
end
