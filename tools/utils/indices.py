import numpy as np
import itertools
import networkx as nx

from more_itertools import powerset
from functools import partial
from scipy.sparse.csgraph import connected_components

# from mage.utils.validation import check_random_state

import torchsnooper

__all__ = [
    'shapley_taylor_indices',
    'myerson_interaction_indices',
]

"""
SHAPLEY_TAYLOR_INDICES
"""


# @torchsnooper.snoop()
def delta_fn(S, T, fn):
    s = len(S)  # 获取集合 S 的长度
    T_set = set(T)  # 将 T 转换为集合
    ret = 0  # 初始化返回值

    for W in powerset(S):  # 遍历 S 的所有子集 W
        w = len(W)  # 获取子集 W 的长度
        value = fn(frozenset(T_set.union(W)))  # 计算函数 fn 在 T 和 W 的并集上的值
        ret += (-1)**(w - s) * value  # 根据子集的长度和集合 S 的长度来更新返回值

    return ret  # 返回计算结果

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.RandomState()
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

@torchsnooper.snoop()
def shapley_taylor_indices(num_players, fn, ord=2, num_samples=500, random_state=None, return_indices=True):
       # Starting var:.. num_players = 25
        # Starting var:.. fn = functools.partial(<function communication_restri...at 0x7fb3ba5563a0>, cpn_dict={}, restricted=True)
        # Starting var:.. ord = 2
        # Starting var:.. num_samples = 200
        # Starting var:.. random_state = 534895718
        # Starting var:.. return_indices = False
    #num_players：玩家数量。fn：需要评估的函数。ord：Shapley-Taylor 指数的阶数。
    # num_samples：用于计算的样本数。random_state：随机种子。return_indices：是否返回指数。
    rng = check_random_state(random_state)#rng = RandomState(MT19937) at 0x7FB3C23DF840
    indices = np.zeros([num_players]*ord, dtype=np.float64)#用于存储计算的 Shapley-Taylor 指数。
    # indices = ndarray<(25, 25), float64>
    sum_inds = np.zeros_like(indices)#存储各子集的增量和。    
    cnt_inds = np.zeros_like(indices)#存储各子集的计数。
    players = np.array(list(range(num_players)))#生成玩家的数组。players = ndarray<(25,), int64>
    print(players)#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]

    for _ in range(num_samples):
        p = np.array(rng.permutation(num_players))#生成玩家的随机排列。p = ndarray<(25,), int64>
        print("p:",p)#p: [ 3 18  4  1 16 20  2 19  7  8  9 22 17 10 24 11  5  0 23 12 15 21 13 14 6]
        inv_p = np.zeros_like(p)#初始化 inv_p 数组为零。
        inv_p[p] = players#生成排列的逆序。为inv_p[p[i]] = players[i]，将 players 数组中的值按照 p 数组的顺序填入 inv_p 数组中
        print("inv_p:",inv_p)#inv_p: [17  3  6  0  2 16 24  8  9 10 13 15 19 22 23 20  4 12  1  7  5 21 11 18 14]

        #计算子集的增量
        for S in itertools.combinations(players, ord):#生成所有可能的大小为 ord 的子集。
            i_k = np.min(inv_p[np.array(S)])#找到子集中最小的逆序索引。
            T = p[:i_k]#选择前 i_k 个元素作为 T。

            delta = delta_fn(S, T, fn)#计算子集 S 和 T 的增量。
            print("delta:",delta)
            # ttt
            if return_indices:#如果 return_indices 为真，更新增量和计数。
                for p_S in itertools.permutations(S):#对 S 的所有排列进行循环。
                    sum_inds[p_S] += delta#更新增量和。
                    cnt_inds[p_S] += 1#更新计数。

    if return_indices:#如果 return_indices 为真，计算平均指数。
        indices = np.divide(sum_inds, cnt_inds, out=np.zeros_like(sum_inds), where=(cnt_inds != 0))
        #将 sum_inds 除以 cnt_inds 计算平均值。
        
    # 处理较小的子集
    for r in range(1, ord):#对所有小于 ord 的子集大小进行循环。
        for S in itertools.combinations(players, r):#生成所有可能的大小为 r 的子集。
            delta = delta_fn(S, tuple(), fn)#计算子集 S 的增量。
            if return_indices:#如果 return_indices 为真，更新增量。
                # Temporary solution, a not good practice
                # We access the index of a subset S size h by indices[T]
                # where T = (S, S[0], S[0], ...S[0]), |S| = h, |T| = ord
                for i in range(len(S)):#对 S 的所有元素进行循环。
                    p_S = S + (S[i],) * (ord - len(S))#将子集 S 扩展到大小为 ord。
                    indices[p_S] = delta#更新指数。

    return indices


def shapley_interaction_indices(num_players, fn, ord=2, num_samples=500,
                                random_state=None, return_indices=True):
    if ord > 2:
        raise NotImplementedError

    rng = check_random_state(random_state)
    indices = np.zeros([num_players]*ord, dtype=np.float32)
    sum_inds = np.zeros_like(indices)
    cnt_inds = np.zeros_like(indices)

    for _ in range(num_samples):
        p = np.array(rng.permutation(num_players))

        for l in range(1, ord+1):
            for k in range(0, num_players - l):
                T = p[:k]
                S = p[k: k+l]

                delta = delta_fn(S, T, fn)

                if return_indices:
                    for p_S in itertools.permutations(S):
                        if len(S) < ord:
                            p_S = (S[0], S[0])
                        sum_inds[p_S] += delta
                        cnt_inds[p_S] += 1

    if return_indices:
        indices = np.divide(sum_inds, cnt_inds, out=np.zeros_like(sum_inds), where=(cnt_inds != 0))

    return indices
