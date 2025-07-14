# Flash Attention

## Flash attention Forward pass

This is the code for FlashAttention:


At first, it looked intimidating, but it's straightforward when you understand it; this is how
I think about it:
1. There are two loops, the outer one is parallel, meaning each thread block will be working on a $B_q$
rows of the overall $Q$ matrix.
2. The inner loop is *sequential*, each thread block will go through all the $T_k$ keys and values, one by one, to finally write the $B_q$ rows of $O$
3. Each time we load a new key and value, we calculate the similarity score between our queries (always loaded) and the newly loaded keys.
4. We update the maximum value of each row with new calculated similarities, we save them into vector $m$.
5. We will use the correction term to recalibrate with the new maximum value
6. We calculate the *un-normalized* similarity matrix $P$ (we add $B_k$ portion at each step of the loop).
7. We accumulate the new similarity matrix entries into the running sum $l$.
8. Finally, we load the rows in the output we’re responsible for, we start by correcting with new max, and then we add the new portion of weighted values.
9. We finish by normalizing the output with the running sum (we factorized the normalization until the end)

> Note: 
> The difference between FlashAttention 1 and this version is that in FlashAttention 1, we parallelize **only** over the *batching dimensions* (batch and head).
> If we implement FlashAttention 1, each block will run the two loops sequentially. Here we parallize the leading dimensions *and* across sequence length; 
> this is how we avoid the quadratic complexity.