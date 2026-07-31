# Verified computing: why, what, how — and who

A primer on the numerical analysis behind *verified computing*: computations that
deliver, along with an answer, a machine-proved bound on how wrong that answer
can be.

This note is the theory companion to the two measurement studies in this repo —
[dot-product-characterization.md](dot-product-characterization.md) and
[interval-blas-study.md](interval-blas-study.md) — which measure on real
arithmetic what is derived here. Where a claim below has been measured, there is
a pointer to the number.

> **On citations.** References are given as author / title / venue / year. Titles
> and years are stated as accurately as I can; volume and page numbers are
> deliberately omitted where I am not certain of them. Verify before quoting this
> document in a paper.

---

## Contents

1. [Why: the problem verified computing solves](#1-why-the-problem-verified-computing-solves)
2. [The vocabulary: forward error, backward error, conditioning](#2-the-vocabulary)
3. [Wilkinson: making rounding error analyzable (1947–1965)](#3-wilkinson-making-rounding-error-analyzable)
4. [The dot product is the bottleneck](#4-the-dot-product-is-the-bottleneck)
5. [Moore: intervals, and why they are not enough alone (1962–1966)](#5-moore-intervals-and-why-they-are-not-enough-alone)
6. [Kulisch: the fifth operation (1970s–2010s)](#6-kulisch-the-fifth-operation)
7. [How verification actually works: defect correction and Krawczyk](#7-how-verification-actually-works)
8. [Higham: making the analysis usable, then realistic (1996–2022)](#8-higham-making-the-analysis-usable-then-realistic)
9. [What we measured](#9-what-we-measured)
10. [Timeline](#10-timeline)
11. [References](#11-references)

---

## 1. Why: the problem verified computing solves

Every floating-point computation returns a number. Almost none of them return
any indication of how much of that number is meaningful.

Consider a concrete failure. Take

$$
x = (10^{16},\; 1,\; -10^{16}), \qquad y = (1,\; 1,\; 1).
$$

The exact inner product is $x^Ty = 1$. Evaluated left to right in IEEE binary64:

$$
\mathrm{fl}(10^{16} + 1) = 10^{16}, \qquad \mathrm{fl}(10^{16} - 10^{16}) = 0.
$$

The computed answer is $0$. Not "1 with a small error" — the answer has **no
correct digits**, and nothing in the result says so. The `1` was *swamped*: added
to a number so much larger that it fell off the bottom of the significand, and
then the large numbers cancelled, leaving only the damage.

Three responses to this are possible, and the history of the field is largely the
history of the third:

1. **Ignore it.** Works more often than it should, which is itself a discovery
   (§3) — but offers no way to know *when* it fails.
2. **Estimate it.** Run the computation twice at different precisions, or perturb
   the input and see how much the output moves. Cheap, informative, and *not a
   proof*: it can be fooled.
3. **Prove it.** Return an interval, or a bound, that is guaranteed to contain
   the exact answer. This is **verified computing**, also called *validated
   numerics* or *result verification*.

The third is what this note is about. Its central difficulty is that a
guaranteed bound is worthless if it is not also **tight**: the interval
$[-\infty, +\infty]$ contains every answer and certifies nothing. Verified
computing is therefore not merely about being correct. It is about being correct
*and* narrow enough to be useful — and, as we will see, the exact dot product is
the single arithmetic primitive that most affects whether that is achievable.

---

## 2. The vocabulary

### 2.1 The model of floating-point arithmetic

Let $u$ be the **unit roundoff**: half the gap between $1$ and the next
representable number. For IEEE binary64, $u = 2^{-53} \approx 1.11 \times
10^{-16}$.

The standard model says every basic operation is correctly rounded:

$$
\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad |\delta| \le u,
\qquad \circ \in \{+, -, \times, \div\}.
$$

This is exactly what IEEE 754 guarantees for $+, -, \times, \div, \sqrt{\cdot}$.
Note carefully what it does **not** cover: a *sequence* of operations, such as a
dot product. That gap is the subject of §4 and §6.

### 2.2 Forward error and backward error

Let $y = f(x)$ be what we want and $\hat{y}$ what we compute.

**Forward error** is the obvious question — how wrong is the answer?

$$
E_{\mathrm{fwd}} = \frac{\|\hat{y} - y\|}{\|y\|}.
$$

**Backward error** is the non-obvious question, and the one that made the field
tractable — *of what nearby problem is our answer the exact solution?*

$$
E_{\mathrm{bwd}} = \min\left\{ \frac{\|\Delta x\|}{\|x\|} \;:\; \hat{y} = f(x + \Delta x) \right\}.
$$

An algorithm is **backward stable** if $E_{\mathrm{bwd}}$ is of order $u$: the
computed answer is the exact answer to a problem indistinguishable from the one
posed.

### 2.3 The condition number, and the rule that ties them together

The **condition number** measures how much the *problem* amplifies input
perturbations — a property of $f$ and $x$, with nothing to do with arithmetic:

$$
\kappa(f, x) = \lim_{\epsilon \to 0} \sup_{\|\Delta x\| \le \epsilon \|x\|}
\frac{\|f(x + \Delta x) - f(x)\|}{\epsilon \|f(x)\|}.
$$

The three quantities are linked by the single most useful rule of thumb in
numerical analysis:

$$
\boxed{\;\text{forward error} \;\lesssim\; \text{condition number} \times \text{backward error}\;}
$$

This separates concerns cleanly, and the separation is the point:

- **Backward error** is the algorithm's responsibility. A good algorithm makes it
  $O(u)$.
- **Condition number** is the problem's property. No algorithm can improve it.
- **Forward error** — what the user actually cares about — is the product, and so
  can be large *even for a perfect algorithm* if the problem is ill-conditioned.

Turing introduced the condition number in this sense in 1948, in "Rounding-off
errors in matrix processes" — the same paper in which he described what we now
call LU factorization.

---

## 3. Wilkinson: making rounding error analyzable

**Purpose:** to explain why numerical linear algebra works in practice, when the
existing theory said it should not.

### 3.1 The problem he inherited

In 1947 von Neumann and Goldstine published "Numerical inverting of matrices of
high order", the first serious rounding-error analysis of matrix computation.
Their bounds were alarming: they suggested that solving large linear systems in
finite precision might be hopeless. Yet practitioners — including Wilkinson,
working with Turing on the ACE at the National Physical Laboratory from 1946 —
kept solving them successfully.

Either the practice was lucky or the theory was pessimistic. Wilkinson spent
fifteen years showing it was the theory.

### 3.2 The insight: backward error analysis

The move that unlocked everything was refusing to track how far the computed
answer drifts from the true one, and instead asking what problem the computed
answer solves *exactly*.

For Gaussian elimination with partial pivoting, Wilkinson's 1961 analysis
("Error analysis of direct methods of matrix inversion") gives: the computed
$\hat{x}$ satisfies

$$
(A + \Delta A)\,\hat{x} = b, \qquad
\|\Delta A\|_\infty \le \gamma_{3n}\, \rho_n \|A\|_\infty,
$$

where $\rho_n$ is the *growth factor* — how much entries swell during
elimination — and $\gamma_n$ is the notation introduced below.

Read that carefully: **the computed solution is the exact solution of a slightly
perturbed system.** The algorithm is therefore not at fault when the answer is
inaccurate; the problem's conditioning is. And since $A$ usually comes from
measurement and is already uncertain at a level far exceeding $u$, a backward
error of order $u$ means the algorithm has done everything that could be asked.

That reframing is why Gaussian elimination is trusted, and it earned Wilkinson
the 1970 Turing Award.

### 3.3 The $\gamma_n$ notation

Wilkinson's bookkeeping device, now universal. When $n$ rounding errors
accumulate multiplicatively,

$$
\prod_{i=1}^{n} (1 + \delta_i)^{\pm 1} = 1 + \theta_n, \qquad
|\theta_n| \le \gamma_n := \frac{n u}{1 - n u}, \qquad (n u < 1).
$$

For $nu \ll 1$, $\gamma_n \approx nu$. The whole apparatus of classical error
analysis is built from this one abbreviation.

### 3.4 Iterative refinement — Wilkinson's other gift

Wilkinson also gave the field its first *cure* rather than diagnosis. Given an
approximate solution $\hat{x}$ of $Ax = b$:

$$
\begin{aligned}
r &= b - A\hat{x} && \text{(residual)} \\
A d &= r && \text{(solve with the existing factorization — cheap)} \\
\hat{x} &\leftarrow \hat{x} + d && \text{(correct)}
\end{aligned}
$$

The classical result, with the residual computed in **extended precision**: if
$\kappa(A) u < 1$, the refined solution attains

$$
\frac{\|\hat{x} - x\|_\infty}{\|x\|_\infty} \approx u,
$$

**independent of $\kappa(A)$**. Compute the residual in working precision instead
and refinement stagnates at $\approx \kappa(A)\,u$.

The reason is exactly the failure of §1. Near convergence $A\hat{x} \approx b$, so
forming $b - A\hat{x}$ is catastrophic cancellation: the leading digits agree and
annihilate, and what survives is precisely what the working precision has already
thrown away. An accurate residual is therefore not a refinement of the method —
it *is* the method.

**Hold onto this.** It is the seed of everything Kulisch built (§6), and we
measure it in §9.4.

---

## 4. The dot product is the bottleneck

### 4.1 Wilkinson's inner product bound

Apply the model of §2.1 to $x^Ty = \sum_{i=1}^n x_i y_i$. Every product rounds,
every partial sum rounds, and the errors accumulate:

$$
\boxed{\;\bigl|\mathrm{fl}(x^Ty) - x^Ty\bigr| \;\le\; \gamma_n \,|x|^T|y| \;}
$$

where $|x|$ denotes componentwise absolute value, so
$|x|^T|y| = \sum_i |x_i||y_i|$.

The bound is also available in backward form,

$$
\mathrm{fl}(x^Ty) = (x + \Delta x)^T y, \qquad |\Delta x| \le \gamma_n |x|,
$$

i.e. the computed dot product is the exact dot product of a slightly perturbed
$x$ — backward stable, in the sense of §2.2.

### 4.2 The crucial detail: the bound is not relative to the answer

The error is bounded by $\gamma_n |x|^T|y|$, **not** by $\gamma_n |x^Ty|$. Those
differ by exactly the amount of cancellation in the sum. Dividing through:

$$
\frac{\bigl|\mathrm{fl}(x^Ty) - x^Ty\bigr|}{|x^Ty|}
\;\le\; \frac{\gamma_n}{2} \cdot \underbrace{\frac{2\,|x|^T|y|}{|x^Ty|}}_{\displaystyle \mathrm{cond}(x,y)}
$$

The quantity

$$
\boxed{\;\mathrm{cond}(x,y) = \frac{2\,|x|^T|y|}{|x^Ty|}\;}
$$

is the **condition number of the dot product**. It has a clean interpretation:
$|x|^T|y|$ is the information the inputs carry in, $|x^Ty|$ is what survives into
the result, and their ratio is what the reduction destroys. $\log_{10}
\mathrm{cond}$ is that loss counted in decimal digits.

It satisfies $\mathrm{cond}(x,y) \ge 2$, with equality exactly when every product
$x_iy_i$ has the same sign. The naming and the factor of $2$ follow Ogita, Rump &
Oishi (2005).

Returning to §1's example: $|x|^T|y| = 2\times10^{16} + 1$ while $|x^Ty| = 1$, so
$\mathrm{cond} \approx 4 \times 10^{16}$ — about 16.6 decimal digits of loss,
against binary64's ~15.95 available. The computed $0$ was not bad luck; it was
arithmetic.

### 4.3 Why *this* operation matters so much

Two reasons, and together they are the whole argument for §6.

**It is everywhere.** Matrix–vector product, matrix–matrix product, norms,
projections, Gram matrices, residuals, orthogonalization — essentially all of
dense and sparse linear algebra is dot products in a trenchcoat. An error model
for the dot product is an error model for numerical linear algebra.

**It is the one composite operation IEEE 754 does not protect.** The standard
guarantees a single rounding for $+,-,\times,\div,\sqrt{\cdot}$. It says nothing
about a *sequence*, and a chain of individually correctly-rounded operations does
**not** compose into a correctly-rounded dot product. The guarantee stops exactly
where linear algebra begins.

---

## 5. Moore: intervals, and why they are not enough alone

**Purpose:** to make the computer carry rigorous bounds automatically, rather
than having a human do error analysis by hand for each program.

Ramon E. Moore's PhD thesis (Stanford, 1962) and book *Interval Analysis* (1966)
founded the field. The idea is disarmingly simple: replace each number by an
interval known to contain it, and define arithmetic so containment is preserved.

$$
[a,b] + [c,d] = [a+c,\; b+d], \qquad
[a,b] - [c,d] = [a-d,\; b-c],
$$
$$
[a,b] \times [c,d] = [\min(ac,ad,bc,bd),\; \max(ac,ad,bc,bd)].
$$

On a real computer each endpoint must additionally be rounded **outward** — the
lower bound toward $-\infty$, the upper toward $+\infty$ — or the guarantee is
lost. The resulting property, the **Fundamental Theorem of Interval
Arithmetic**, is what makes the whole thing worth doing:

$$
\boxed{\;\forall\, x \in X,\; y \in Y: \quad x \circ y \;\in\; \mathrm{fl}(X \circ Y)\;}
$$

evaluated over the exact reals. Compose operations and containment composes: the
final interval provably contains the exact answer.

### 5.1 Three ways it goes wrong

Interval arithmetic is *sound* by construction. Being **useful** is another
matter, and there are exactly three ways to lose it. Distinguishing them is
essential, because they have different cures — and only one of them is cured by
better arithmetic.

**(a) Accumulation.** Each operation rounds outward, so a length-$n$ reduction
widens $n$ times. The enclosure ends up tracking the *error bound* $\gamma_n
\mathrm{cond}$ rather than the actual uncertainty. **Curable** — this is what §6
is for.

**(b) The dependency problem.** Interval arithmetic treats every occurrence of a
variable as independent. So for $X = [1,2]$,

$$
X - X = [1-2,\; 2-1] = [-1, 1] \;\ne\; [0,0],
$$

and for $X = [-1,2]$,

$$
X \cdot X = [-2, 4] \quad \text{whereas} \quad X^2 = [0, 4].
$$

The interval is correct but needlessly wide, and **no accumulator fixes this** —
an exact accumulator will faithfully and exactly accumulate the wrong corner
products. The cure is to rewrite the *expression*.

**(c) The wrapping effect.** A set that is not an axis-aligned box must be stored
as one. Rotate a box by $\theta$ and take the axis-aligned hull: the width is
multiplied by

$$
|\cos\theta| + |\sin\theta| \;\in\; [1, \sqrt{2}],
$$

peaking at $\theta = 45°$. Apply $k$ rotations and the overestimation compounds
as $(|\cos\theta|+|\sin\theta|)^k$ — *geometric growth*, from a transformation
that is exactly norm-preserving. This is a loss in the **representation**, not
the arithmetic, and **no accumulator and no re-expression** fixes it.

The rest of this note is largely about (a), because (a) is the one that yields.

---

## 6. Kulisch: the fifth operation

**Purpose:** to make automatic result verification *achievable*, by fixing the one
arithmetic gap that makes rigorous bounds too wide to use.

Ulrich Kulisch, at Karlsruhe from the 1970s onward, made an argument that is
easier to state than it was to win:

> A computer implements $+,-,\times,\div$ with a single rounding of the exact
> result. It should implement the **dot product** the same way. That is a *fifth
> basic arithmetic operation*, and it should be in the hardware.

### 6.1 The argument, in four steps

1. IEEE 754 correctly rounds the four operations: one rounding, exact result
   rounded once.
2. The dot product is the operation from which linear algebra is built (§4.3),
   and it is **not** among them. Chaining correctly-rounded operations does not
   yield a correctly-rounded dot product.
3. Therefore add the **exact dot product** (EDP) as a fifth operation:

$$
\boxed{\;\mathrm{EDP}(x,y) = \mathrm{round}\!\left(\sum_{i=1}^{n} x_i y_i\right)\;}
$$

   with the sum formed *exactly* and rounded exactly once at the end. The error is
   then $\le u$ **relative to the result**, with no $\gamma_n$ and no
   $\mathrm{cond}$ — compare against §4.2:

$$
\underbrace{\frac{\gamma_n}{2}\,\mathrm{cond}(x,y)}_{\text{conventional}}
\qquad\text{versus}\qquad
\underbrace{u}_{\text{EDP}}
$$

4. The payoff is not aesthetic. An exact residual makes **defect correction**
   effective (§3.4, §7), and defect correction plus interval arithmetic yields
   **verified enclosures** — automatically.

### 6.2 The implementation: a complete register

The insight that makes it practical is that the cost is *bounded*. Because a
floating-point format has a bounded exponent range, every product $x_i y_i$ lives
between $\mathrm{minpos}^2$ and $\mathrm{maxpos}^2$. A fixed-point accumulator
spanning that range, plus a few guard bits for carries, can hold **any** sum of
**any** number of such products with no rounding at all:

$$
\text{width} \;\approx\; \underbrace{2|e_{\min}| + 2f}_{\text{below the point}}
\;+\; \underbrace{2 e_{\max}}_{\text{above}} \;+\; \underbrace{k}_{\text{carry guard}}
\quad\text{bits.}
$$

For binary64 this is on the order of a couple of thousand bits — large by
register standards, trivial by memory standards, and **fixed**: it does not grow
with $n$. Kulisch called it the *complete register*; posit arithmetic calls it
the **quire**; the general pattern is a *super-accumulator*.

Accumulation into it is integer addition — no normalization, no rounding, and
(usefully) **associative**, so the result is independent of summation order and
therefore reproducible across parallel schedules.

### 6.3 What Kulisch was actually after

It is worth being precise, because the modern framing often gets this backwards.

The EDP is frequently motivated today by **reproducibility** — bit-identical
results across thread counts and machines. That is a genuine benefit and it
follows for free from exactness. But it is a *later* re-motivation, adopted
roughly two decades after the fact, and it is not what Kulisch was arguing for.

His goal was **validity**: computations whose results come with machine-proved
enclosures, produced automatically rather than by hand analysis. The line of
reasoning is

$$
\text{exact dot product}
\;\Longrightarrow\; \text{accurate residual}
\;\Longrightarrow\; \text{effective defect correction}
\;\Longrightarrow\; \text{verified enclosure}.
$$

This shows in what he built: the ACRITH library for IBM System/370 (early
1980s), and the *XSC* language family (PASCAL-XSC, C-XSC, ACRITH-XSC) — all of
them verification systems, not reproducibility systems. The titles say it too:
*Computer Arithmetic in Theory and Practice* (with Miranker, 1981), "The
Arithmetic of the Digital Computer: A New Approach" (SIAM Review, 1986), and
*Computer Arithmetic and Validity* (2013).

The reproducibility line — ReproBLAS (Demmel, Ahrens & Nguyen), ExBLAS
(Iakymchuk, Collange, Defour & Graillat) — arrived later, when large parallel
machines made summation order nondeterministic. Gustafson & Yonemoto's posit
proposal (2017) reintroduced the same structure as the **quire**, this time as a
mandatory part of the number system rather than an optional library.

---

## 7. How verification actually works

Knowing that the EDP helps is not the same as knowing how a verified solve is
built. Here is the machinery.

### 7.1 What does *not* work: interval elimination

The naive approach — run Gaussian elimination with interval entries — fails badly.
Pivots become intervals containing zero, division by them is unbounded, and widths
explode. For anything but small, well-conditioned, strongly diagonally dominant
systems the result is $[-\infty, +\infty]$: sound, and useless.

This is not a defect to be engineered around. It is the accumulation and
dependency problems of §5.1 compounding over $O(n^3)$ operations.

### 7.2 What works: verify an approximate solution

The productive idea — due in this form to Krawczyk (1969) and developed
extensively by Rump — is to *not* compute in intervals at all. Instead:

1. Compute an approximate solution $\tilde{x}$ in ordinary floating point. Fast.
2. Compute an approximate inverse $R \approx A^{-1}$, also in floating point.
   $R$ need not be accurate; it is a *preconditioner*.
3. Then **prove**, with interval arithmetic used only for the proof, that the
   true solution lies near $\tilde{x}$.

Define the residual and the iteration matrix

$$
z = R\,(b - A\tilde{x}), \qquad C = I - RA .
$$

$C$ measures how far $R$ is from a true inverse. The **Krawczyk operator** is

$$
K(X) = z + C\,X ,
$$

and the theorem is:

$$
\boxed{\;K(X) \subseteq \mathrm{int}(X)
\;\Longrightarrow\;
A \text{ is nonsingular, and } x^* \in \tilde{x} + X. \;}
$$

A single inclusion test proves both **existence** and **uniqueness** of the
solution, and bounds it. When $\|C\|_\infty < 1$ the enclosure radius satisfies

$$
\|X\|_\infty \;\lesssim\; \frac{\|z\|_\infty}{1 - \|C\|_\infty}.
$$

### 7.3 Where the exact dot product enters

Look at what governs each factor:

- $\|C\|_\infty < 1$ is the **feasibility** condition. It depends on the quality
  of the preconditioner $R$ — i.e. on $\kappa(A)$ and the precision of the
  inversion. The EDP does not help here.
- $\|z\|_\infty$ sets the **tightness**. And $z$ is a preconditioned *residual*,
  computed at the point where $A\tilde{x} \approx b$ — catastrophic cancellation,
  $\mathrm{cond}$ of order $\kappa(A)$.

Computed in working precision, $z$ is dominated by its own rounding error, and
the enclosure width degrades as $\kappa(A) u$ no matter how good $\tilde x$ was.
Computed exactly, $z$ reflects the true residual and the enclosure is as tight as
$\tilde{x}$ deserves.

**That is Kulisch's argument, localized to one expression.** The exact dot
product does not make the solver better; it makes the *proof about* the solver
tight enough to be worth having.

---

## 8. Higham: making the analysis usable, then realistic

**Purpose:** first to turn error analysis from folklore into an engineering
discipline; later, to make its bounds honest for modern low-precision hardware.

### 8.1 Systematization (1996, 2002)

By the 1990s, backward error analysis was fifty years old and scattered across
hundreds of papers, each with its own notation. Nicholas J. Higham's *Accuracy
and Stability of Numerical Algorithms* (1996; 2nd ed. 2002) collected it into one
coherent treatment — consistent notation, worked analyses for essentially every
standard algorithm, and, importantly, honest discussion of when the bounds are
tight and when they are wildly pessimistic.

The practical effect is hard to overstate: the bounds in §3 and §4 are quoted in
this note in Higham's notation because that is the notation the field now uses.
Chapter 3 covers dot products and chapter 4 summation; the inner-product bound of
§4.1 is Higham's statement of Wilkinson's result.

ASNA also documents a fact worth knowing when designing experiments: **triangular
systems are typically solved to much higher accuracy than their condition number
suggests** (ch. 8). We rediscovered this the hard way — see §9.4.

### 8.2 Realism: the $\sqrt{n}$ result (2019)

The classical constant $\gamma_n \approx nu$ is a **worst case**, and it assumes
every rounding error conspires to push the same direction. For $n = 10^6$ in
binary64 it predicts a relative error near $10^{-10}$ — but measured errors are
routinely orders of magnitude smaller.

Higham & Mary, "A New Approach to Probabilistic Rounding Error Analysis" (2019),
replaced the worst case with a probabilistic model: treat the $\delta_i$ as
independent mean-zero random variables. Then errors accumulate like a random walk
rather than a sum, and $\gamma_n$ is replaced by

$$
\tilde{\gamma}_n(\lambda) \;=\; \exp\!\left(\frac{\lambda\sqrt{n}\,u + n u^2}{1-u}\right) - 1
\;\approx\; \lambda \sqrt{n}\, u ,
$$

holding with a probability controlled by $\lambda$. The deterministic $n$ becomes
$\sqrt{n}$ — a difference of $1000\times$ at $n = 10^6$.

**Why this mattered when it did.** Machine learning pushed fp16 and bfloat16 into
production, where $u \approx 10^{-3}$ and $nu > 1$ for quite modest $n$ — the
classical bound becomes *vacuous*, predicting no correct digits for computations
that work fine. A realistic bound was no longer a refinement but a necessity.

An important caveat: the probabilistic model assumes errors are independent and
mean-zero. For a **one-signed** reduction — a sum of squares, say — the errors do
not cancel, and the pessimistic $nu$ growth is real. Not every reduction gets the
$\sqrt{n}$.

Blanchard, Higham & Mary (2020) extended this to blocked and pairwise summation,
where growth is $O(\log n)$ instead.

### 8.3 Mixed precision (2017–2022)

The third phase connects directly back to Wilkinson. With hardware offering
several precisions at wildly different speeds, the question becomes: which
precision does each *part* of an algorithm need?

Carson & Higham (2017, 2018) analyzed three-precision iterative refinement —
factor in low precision, solve in working precision, compute the residual in high
precision — and characterized exactly when it converges. This is Wilkinson's 1963
scheme, now with the precisions as free parameters, and with the modern insight
that the *factorization* can be much cheaper than the working precision as long
as the *residual* is more accurate. Higham & Mary's *Acta Numerica* survey (2022)
is the synthesis.

Note the shape of the conclusion: **the residual is the part that must be
accurate.** That is Kulisch's argument arrived at from a completely different
direction — economics rather than axiomatics.

---

## 9. What we measured

Theory is more convincing with numbers attached. Everything below was measured on
posit and cfloat arithmetic in this repository; the pointers go to the full
tables.

### 9.1 `cond` predicts the error, over twenty decades

Sweeping $\mathrm{cond}(x,y)$ from $10^3$ to $10^{23}$ at fixed $n = 4096$
([characterization §4.2](dot-product-characterization.md)):

| accumulator | measured relative error |
|---|---|
| native `posit<32,2>` | $\approx 1.4\times10^{-9} \cdot \mathrm{cond}$ |
| promoted `double` | $\approx 8.1\times10^{-18} \cdot \mathrm{cond}$ |
| exact quire | $0$, flat across all twenty decades |

The $\mathrm{cond}$ dependence of §4.2 is not a loose bound; it is an equality up
to a small constant. And the EDP's independence from it is exact.

### 9.2 `cond` is necessary but not sufficient

An all-positive reduction has $\mathrm{cond} = 2$ — the best possible — and is
still **25% wrong** in `posit<16,2>` at $n = 4096$, because $n_{\text{eff}}$, the
count of terms still large enough to move the accumulator, falls to 1723 of 4096.
That is *swamping* (§1), which $\mathrm{cond}$ cannot see: it is the $\gamma_n$
factor, not the $\mathrm{cond}$ factor. Report both.

### 9.3 The $\sqrt{n}$ result, and a caveat for tapered formats

Holding $\mathrm{cond}$ fixed and sweeping $n$ over a 256× range
([characterization §4.3](dot-product-characterization.md)):

- a **fixed-precision** `double` accumulator grew $5.6\times$ — consistent with
  Higham & Mary's $\sqrt{n}$ (which predicts $16\times$) and refuting the
  classical $n$ (which predicts $256\times$);
- a **tapered-precision** posit accumulator grew $\approx 1300\times$ —
  *superlinearly*, because posit's local unit roundoff coarsens as the running
  sum moves away from $1$. No constant-$u$ model covers this. For posit, "unit
  roundoff" is not a single number.

### 9.4 The verified solve, and a lesson about experimental design

Defect correction on a Hilbert system in `posit<32,2>` ($u = 3.7\times10^{-9}$),
relative forward error after four steps
([interval study §12](interval-blas-study.md)):

| $n$ | $\kappa$ | working-precision residual | **exact residual** |
|---:|---:|---:|---:|
| 4 | $2.8\times10^{4}$ | $1.8\times10^{-5}$ *(worse than it started)* | $\mathbf{3.7\times10^{-9}} = u$ |
| 6 | $2.9\times10^{7}$ | $2.1\times10^{-2}$ | $\mathbf{3.0\times10^{-9}} \approx u$ |

Raising $\kappa$ by $1000\times$ leaves the exact-residual accuracy **unchanged**
at $u$. That is §3.4 and §6.3 measured — accuracy decoupled from conditioning.

Past $\kappa u \approx 1$ neither residual helps: the *factorization* has no
correct digits, and the EDP removes the $\kappa$ dependence of the **residual**,
not of the factorization. The claim is bounded.

**The design lesson.** Our first attempt used triangular systems, and found *no
difference at all* between the two residuals. The reason is §8.1: triangular
solves are far more accurate than their condition number suggests, so
back-substitution lands within a few ulp and defect correction has nothing to
correct. An experiment built on `trsv` would have passed vacuously and concluded
the exact residual does not help — the most dangerous kind of wrong result. Read
the literature before designing the benchmark.

### 9.5 The three limits, and what the EDP is actually for

Measuring each of §5.1's failure modes separately
([interval study §13](interval-blas-study.md)):

| obstacle | example | exact accumulator | re-expression |
|---|---|---|---|
| **accumulation** | `dot`, `gemv`, `gemm`, residual | **fixes it** | — |
| **interface** | a reduction spread across calls | not alone | **fixes it** |
| **dependency** | $\|x\|_2$ as $\sqrt{x^Tx}$ | no | **fixes it** |
| **wrapping** | repeated plane rotation | no | no |

The headline number: on an adversarial reduction at $\mathrm{cond} = 2.7 \times
10^{19}$, naive interval arithmetic and a promoted-`double` interval accumulator
both certify a **negative** number of digits — enclosures wider than the value
they enclose, proving nothing — while the exact quire certifies **15.4 digits**.

So the honest summary of what Kulisch's fifth operation buys is narrow and
specific:

> **The exact dot product makes reductions exact.** Everything it buys follows
> from that one property, and everything it cannot do falls outside it.

That is a considerably sharper claim than "the exact dot product makes interval
arithmetic work" — and it is the one the measurements support.

---

## 10. Timeline

| year | who | what | problem being solved |
|---|---|---|---|
| 1947 | von Neumann & Goldstine | first rounding-error analysis of matrix inversion | is finite-precision linear algebra viable at all? |
| 1948 | Turing | "Rounding-off errors in matrix processes"; **condition number** | separating problem sensitivity from algorithm quality |
| 1961 | **Wilkinson** | backward error analysis of Gaussian elimination | why does elimination work when theory says it shouldn't? |
| 1962–66 | Moore | **interval analysis** | automatic rigorous bounds without hand analysis |
| 1963 | **Wilkinson** | *Rounding Errors in Algebraic Processes*; $\gamma_n$; iterative refinement with extended-precision residuals | a usable calculus of rounding error |
| 1965 | **Wilkinson** | *The Algebraic Eigenvalue Problem* | the same, for eigenproblems |
| 1969 | Krawczyk | the Krawczyk operator | proving existence + uniqueness by an inclusion test |
| 1970 | — | Wilkinson receives the Turing Award | — |
| 1981 | **Kulisch** & Miranker | *Computer Arithmetic in Theory and Practice* | axiomatizing computer arithmetic; the complete register |
| ~1983 | **Kulisch** (IBM) | ACRITH for System/370 | verified computing in a shipping product |
| 1986 | **Kulisch** & Miranker | "The Arithmetic of the Digital Computer: A New Approach" (SIAM Review) | the EDP as a fifth basic operation |
| 1980s–90s | **Kulisch** et al. | PASCAL-XSC, C-XSC, ACRITH-XSC | verification as a language feature |
| 1996, 2002 | **Higham** | *Accuracy and Stability of Numerical Algorithms* | making error analysis an engineering discipline |
| 2005 | Ogita, Rump & Oishi | compensated `Sum2`/`Dot2`; $\mathrm{cond}(x,y)$ | doubled accuracy without wide accumulators |
| 2010 | Rump | "Verification methods" (*Acta Numerica*) | the modern synthesis of validated numerics |
| 2013 | **Kulisch** | *Computer Arithmetic and Validity* (2nd ed.) | the mature statement of the programme |
| 2015 | IEEE | Std 1788, interval arithmetic | standardizing containment semantics |
| 2016–17 | Demmel et al.; Iakymchuk et al. | ReproBLAS, ExBLAS | reproducibility on parallel machines |
| 2017 | Gustafson & Yonemoto | posit arithmetic, with the **quire** | the EDP as a mandatory part of the format |
| 2017–18 | Carson & **Higham** | three-precision iterative refinement | which precision does each phase need? |
| 2019 | **Higham** & Mary | probabilistic rounding error analysis; $\sqrt{n}u$ | bounds that are not vacuous at low precision |
| 2020 | Blanchard, **Higham** & Mary | fast and accurate summation algorithms | $O(\log n)$ growth via blocking |
| 2022 | **Higham** & Mary | mixed precision in numerical linear algebra (*Acta Numerica*) | the modern synthesis |

### The through-line

Read the table as one argument developing over seventy-five years:

- **Wilkinson** asked *how wrong is it, and why is that acceptable?* — and gave
  the field backward error analysis, plus the observation that an accurate
  residual makes correction possible.
- **Kulisch** asked *what would the hardware have to provide for the bound to be
  provable and tight?* — and answered: one more operation, exactly rounded.
- **Higham** asked *what is actually true in practice?* — and made the bounds
  usable, then realistic, then precision-aware.

They are three answers to the same question at three different levels: analysis,
architecture, and engineering practice.

---

## 11. References

**Foundational**

- J. von Neumann and H. H. Goldstine, "Numerical inverting of matrices of high order", *Bulletin of the AMS*, 1947.
- A. M. Turing, "Rounding-off errors in matrix processes", *Quarterly Journal of Mechanics and Applied Mathematics*, 1948.

**Wilkinson**

- J. H. Wilkinson, "Error analysis of direct methods of matrix inversion", *Journal of the ACM*, 1961.
- J. H. Wilkinson, *Rounding Errors in Algebraic Processes*, HMSO / Prentice-Hall, 1963.
- J. H. Wilkinson, *The Algebraic Eigenvalue Problem*, Oxford University Press, 1965.

**Interval analysis and verification**

- R. E. Moore, *Interval Analysis*, Prentice-Hall, 1966.
- R. Krawczyk, "Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken", *Computing*, 1969.
- S. M. Rump, "Verification methods: Rigorous results using floating-point arithmetic", *Acta Numerica*, 2010.
- IEEE Std 1788-2015, *IEEE Standard for Interval Arithmetic*.

**Kulisch**

- U. Kulisch and W. L. Miranker, *Computer Arithmetic in Theory and Practice*, Academic Press, 1981.
- U. Kulisch and W. L. Miranker, "The Arithmetic of the Digital Computer: A New Approach", *SIAM Review*, 1986.
- U. Kulisch, *Computer Arithmetic and Validity: Theory, Implementation, and Applications*, de Gruyter, 2nd ed., 2013.

**Higham**

- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, SIAM, 2nd ed., 2002.
- E. Carson and N. J. Higham, "Accelerating the solution of linear systems by iterative refinement in three precisions", *SIAM J. Sci. Comput.*, 2018.
- N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding Error Analysis", *SIAM J. Sci. Comput.*, 2019.
- P. Blanchard, N. J. Higham, T. Mary, "A Class of Fast and Accurate Summation Algorithms", *SIAM J. Sci. Comput.*, 2020.
- N. J. Higham and T. Mary, "Mixed precision algorithms in numerical linear algebra", *Acta Numerica*, 2022.

**Accurate and reproducible summation**

- T. Ogita, S. M. Rump, S. Oishi, "Accurate Sum and Dot Product", *SIAM J. Sci. Comput.*, 2005.
- J. Demmel, P. Ahrens, H. D. Nguyen, "Efficient Reproducible Floating Point Summation and BLAS" (ReproBLAS), 2016.
- R. Iakymchuk, S. Collange, D. Defour, S. Graillat, "ExBLAS: Reproducible and Accurate BLAS Library", 2015.
- J. Gustafson and I. Yonemoto, "Beating Floating Point at its Own Game: Posit Arithmetic", *Supercomputing Frontiers and Innovations*, 2017.

---

## Related in this repository

- [dot-product-characterization.md](dot-product-characterization.md) — the
  structural feature vector ($\mathrm{cond}$, $n_{\text{eff}}$, growth, sign
  balance) and the accuracy/reproducibility measurements of §9.1–9.3.
- [interval-blas-study.md](interval-blas-study.md) — the interval BLAS: enclosure
  tightness across levels 1–3, the three limits of §5.1 measured separately, and
  the verified solve of §9.4.
- [level1-accumulator-study.md](level1-accumulator-study.md) — accumulator width
  versus reduction length, the empirical starting point for all of the above.
