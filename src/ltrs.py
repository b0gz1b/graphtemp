import numpy as np

def find_intersection(interval1, interval2):
    a, b = interval1
    c, d = interval2

    # Check if there is an intersection
    if b < c or d < a:
        return None
    
    # Calculate the endpoints of the intersection
    start = max(a, c)
    end = min(b, d)

    return (start, end)

def not_in_interval(a, interval):
    return [a[i] for i in range(len(a)) if not (interval[0] <= a[i] <= interval[1])]

def update_intervals(interval_list, index_of_change):
    na, nb = interval_list[index_of_change]
    res = interval_list.copy()
    for ind in range(index_of_change+1, len(interval_list)):
        i = ind - index_of_change
        ni = nb + i
        if interval_list[ind][1] < ni:
            return None
        if interval_list[ind][0] < ni:
            res[ind] = (ni,interval_list[ind][1])
    for i in range(1,index_of_change+1):
        ind = index_of_change - i
        ni = na - i
        if interval_list[ind][0] > ni:
            return None
        if interval_list[ind][1] > ni:
            res[ind] = (interval_list[ind][0],ni)
    return res

def temp_cost(p):
    tau = 0
    for s in p:
        tau += len(s)
    return tau

def collapse(p):
    res = []
    for l in p:
        chosen = set()
        for a,b in l:
            if a == b or a != -np.inf:
                chosen.add(a)
            elif b != np.inf:
                chosen.add(b)
            else:
                return None
        res.append(chosen)
    return(res)

def flatten(xss):
    return [x for xs in xss for x in xs]

def longest_time_compatible_subsequence(arr):
    n = len(arr)
    dp = [[1 for _ in range(len(arr[i]))] for i in range(n)]
    longest_ss = [[[o] for o in arr[i]] for i in range(n)]
    longest_ss_index = [[[i] for _ in arr[i]] for i in range(n)]

    for i in range(1, n):
        for k in range(len(arr[i])):
            for j in range(i):
                for l in range(len(arr[j])):
                    if arr[i][k] > arr[j][l] and i - j <= arr[i][k] - arr[j][l]:
                        if dp[j][l] + 1 > dp[i][k]:
                            dp[i][k] = dp[j][l] + 1
                            longest_ss[i][k] = longest_ss[j][l] + [arr[i][k]]
                            longest_ss_index[i][k] = longest_ss_index[j][l] + [i]

    max_length = max(flatten(dp))
    longest_ss_list = []
    longest_ss_index_list = []
    for ss,ssi in zip(longest_ss, longest_ss_index):
        for s,si in zip(ss, ssi):
            if len(s) == max_length:
                longest_ss_list.append(s)
                longest_ss_index_list.append(si)
    possible_labellings = []
    for ss, ssi in zip(longest_ss_list,longest_ss_index_list):
        labelling = []
        j = 0
        for i in range(n):
            if j == max_length:
                labelling.append((ss[j-1] + i - ssi[j-1], np.inf))
            elif i != ssi[j]:
                if j == 0:
                    labelling.append((-np.inf, ss[j] - ssi[j] + i))
                else:
                    labelling.append((ss[j-1] + i - ssi[j-1], ss[j] + i - ssi[j]))
            else:
                labelling.append((ss[j], ss[j]))
                j += 1
        possible_labellings.append(labelling)
    return longest_ss_list, possible_labellings

def connect_chain(arr):
    lss1,pl_1 = longest_time_compatible_subsequence(arr)
    lss2,pl_2 =  longest_time_compatible_subsequence(arr[::-1])
    pl_2_rev = [p[::-1] for p in pl_2]
    lss2_rev = [p[::-1] for p in lss2]
    p_cs = []
    for p1 in pl_1:
        for p2 in pl_2_rev:
            p_c1 = []
            p_c2 = []
            i_inter = None
            for i, (i1, i2) in enumerate(zip(p1,p2)):
                inter = find_intersection(i1,i2)
                if inter is None or i_inter is None:
                    p_c1.append(i1)
                    p_c2.append(i2)
                else:
                    p_c1.append(inter)
                    p_c2.append(inter)
                    i_inter = i
            
            if i_inter != None:
                p_c1 = update_intervals(p_c1, i_inter)
                if None in p_c1:
                    continue
                p_c2 = update_intervals(p_c2[::-1], len(p_c2) - 1 - i_inter)[::-1]
                if None in p_c2:
                    continue
            p_c = list(zip(p_c1,p_c2))
            for i, i_s in enumerate(p_c):
                missing = []
                for s in i_s:
                    missing.append(set(not_in_interval(arr[i], s)))
                m_s = missing[0]
                for m in missing:
                    m_s = m_s.intersection(m)
                p_c[i]= (p_c[i][0],p_c[i][1]) + tuple((m,m) for m in m_s)
            p_cs.append((i_inter,p_c))
    # if found_inter:
    #     p_cs = [p_c for i,p_c in p_cs if i != None]
    # else:
    np_cs = []
    best = np.inf
    for _,p_c in p_cs:
        p_ci = collapse(p_c)
        a = temp_cost(p_ci)
        if a == best:
            np_cs.append(p_ci)
        elif a < best:
            best = a
            np_cs = [p_ci]
    return np_cs
                    

arr = list(map(list,map(set,np.random.randint(1,15,size=(8,4)))))
print(arr)
p_cs = connect_chain(arr)

for p_c in p_cs:
    print(p_c, temp_cost(p_c) - temp_cost(arr))