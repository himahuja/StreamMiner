import heapq

def relax(weight, u, v, r, Dist, prev):
    d = Dist.get(u, inf) + weight
    if d < Dist.get(v, inf):
        Dist[v] = d
        prev[v] = (-weight, u, r)

def get_shortest_path(G, sid, pid, oid):
    #making sure that nodes are integers:
    sid = int(sid)
    oid = int(oid)
    #prev is of the type: [weight, node, relation]
    Dist, visited, priority_q, prev = {sid:0}, set(), [(0,sid)], {sid:(0, -1, -1)}
    path_stack, rel_stack, weight_stack = [], [], []
    while priority_q:
        _, u = heapq.heappop(priority_q)
        if u == oid:
            k = u
            path_stack = [oid]
            rel_stack = [prev[oid][2]]
            weight_stack = [prev[oid][0]]
            while prev[k][1] != -1:
                path_stack.insert(prev[k][1], 0)
                rel_stack.insert(prev[k][2], 0)
                weight_stack.insert(prev[k][0], 0)
                k = prev[k][1]
            path_stack.insert(sid, 0)
        if u in visited:
            continue
        visited.add(u)
        # get the neighbours and cost of the node u
        # returns [relations, neighbors, cost]
        rels, nbrs, costs = G.get_neighbors_sm_packed(u)
        for rel, nbr, cost in zip(rels, nbrs, costs): # for the iteration through keys
            if cost != 0:
                relax(-cost, u, nbr, rel, Dist)
                heapq.heappush(priority_q, (-cost, nbr))
    disceovered_path = RelationalPathSM(s, p, o, 0., len(path_stack)-1, path_stack, rel_stack, weight_stack)
    return discovered_path
