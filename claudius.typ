Witout loss of generality let $pi_(n -> m): bb(R)^(n times n) -> RR^(m times m)$ with $m < n$ be the operator that projects a $n times n$ matrix onto a $m times m$ matrix, by simply taking the first $m$ rows and columns. #footnote()[This is essentially what is done when doing `eora_mario.w.loc[to_region_idx, to_region_idx]`]. 

Let $bold(L)_n: bb(R)^(n times n) -> bb(R)^(n times n)$ be the Leontief operator, that maps a matrix $A$ to it's Leontief matrix, so $ bold(L)_n A = (I - A)^(-1) = sum_(i=0)^infinity A^i. $ 
The series is often called *Neumann series*.

Let $A in bb(R)^(n times n)$ be our technical coefficient matrix arising from empirical data.
Dependency shares are defined as followed in the appendix and are also calculated like this in the replication code

$ "DS" = (v_s (pi_(n ->s ) (bold(L)_n A)) (y_(n -> s) + z_(n -> s))) / (|| v_s ||_1 ). $

The issue lies in the way the Leontief matrix is chosen and the role of intermediate demand, more concretly $ pi_(n -> s) (bold(L)_n A) != bold(L)_s (pi_(n -> s) A) $

This fact can be easily seen by looking at the *Neumann Series*.

The problem is, that $pi_(n->s) bold(L)_n A$ also captures indirect demand from sectors $j > s$.
This can be seen by looking at the final demand.

Let $ tilde(pi)_(n -> s): bb(R)^n -> bb(R)^s $ 
be the projection that takes the first $s$ entries of a vector, Then

$ tilde(pi)_(n->s) ((L_n A)_(n s) tilde(pi)_(n->s) y) =  (pi_(n_s) (L_n A))( tilde(pi)_(n->s) y). $
This can be easily verified by looking at the definition of matrix multiplication.

So it does not matter if the full leontief matrix is chosen or just a submatrix to get the souths demand, when first calculalting the Leontief with the full technological coefficient matrix. This means all information about intermediates is already in the leontief coefficients and intermediates from the north are already captured.

Instead the following approach has to be chosen.

Total production is given as

$ y_i = sum_(j=1)^s a_(i->j) y_i + sum_(j=s+1)^n x_(i->j) + c_i $ 

for all $i < n,$ or:
$ y = A_(s s) y + z + c \ 
  z + c =  underbrace((I -A_(s s)), L pi_(n-> s) A) y. \
  y = (I - A_(s s))^(-1)(z + c)
$

Thus the correct definition of the total demand shares should be either  

$ "DS" = (v_s (pi_(n ->s ) (bold(L)_n A)) (y_(n -> s) )) / (|| v_s ||_1 ). $

or


$ "DS" = (v_s ( (bold(L)_s pi_(n ->s )A)) (y_(n -> s) + z_(n->s) )) / (|| v_s ||_1 ). $

