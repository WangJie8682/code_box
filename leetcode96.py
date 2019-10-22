def numTrees(n):
    #return int
    dp = [0]*(n+1)
    dp[0],dp[1] = 1,1
    for i in range(2,n+1):
        for j in range(1,i+1):
            dp[i]+=dp[j-1]*dp[i-j]
    return dp[-1]


def numTrees(n):
    if n<=1:
        return 1
    sum = 0
    for i in range(1,n+1):
        left = numTrees(i-1)
        right = numTree(n-i)
        sum+=left*right
    return sum


def generateTree(n):
    if n<=0:
        return creat(1,0)
    return creat(1,n)
def creat(low,high):
    res = []
    if low>high:
        return res
    for i in range(low,high+1):
        left = creat(low,i-1)
        right = creat(i+1,high)
        for k in left:
            for j in right:
                root = TreeNode(i)
                root.left = k
                root.right = j
                res.append(root)
    return res
