// Content of the §6 tutorial: one entry per step. Formulas follow paper/sections/06_inference_scaling.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), t = S('t'), u = S('u'), pd = S('pd'), h = S('h'), lam = S('lam'), xh = S('xh'), r = S('r');
const N = S('N'), pN = S('pN'), Fr = S('Fr'), beta = S('beta'), ps = S('ps'), z = S('z'), X = S('X');
const y = S('y'), sy = S('sy'), zeta = S('zeta'), G = S('G'), rh = S('rh');

export const INTRO = `
  <div class="kicker">Survey §6 · Interactive tutorial</div>
  <h1>Inference-time scaling</h1>
  <p>A trained diffusion or flow model can be made better without training it further, by spending more network evaluations when it samples. This tutorial counts that compute in evaluations of the field (NFE) and asks what each extra evaluation buys. It starts with the cheapest lever, a better ODE solver. It then spends compute on many samples and a reward: best-of-N, a search over the starting noise, gradient-based optimization through the sampler, and sequential Monte Carlo (SMC), which reweights and resamples a population along the way. Step 5 conditions on a measurement instead of a reward, which is the inverse-problem setting of DPS. The last step asks what happens when the reward used for selection is itself wrong.</p>
  <p>The base model is the exact field of the three-mode mixture of §2.1 in steps 1, 2, 6 and 7, and of the 2D mixture of §2.2 in steps 3 to 5, so the base model has no error of its own. Steps 1, 2, 4 and 5 are exact up to the solver and the grid. Steps 3, 6 and 7 are Monte Carlo over seeded runs, and each plot says which of its curves are estimated. The convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\).</p>
  <ol class="toc">
    <li><a href="#solvers">Better solvers first</a></li>
    <li><a href="#bestofn">Best-of-N</a></li>
    <li><a href="#search">Searching the noise</a></li>
    <li><a href="#flowgrad">Optimizing through the flow</a></li>
    <li><a href="#dps">Inverse problems</a></li>
    <li><a href="#smc">SMC against best-of-N</a></li>
    <li><a href="#verifier">An imperfect verifier</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'solvers', kicker: 'The cheapest lever', title: 'Better solvers first', scene: 'solvers',
    prose: [
      'Before any sample is thrown away, the compute of one sample can be spent better. The Euler method moves along the velocity at the start of each step and has a global error of order h. Heun takes an Euler step, evaluates the field at the end, and moves along the average of the two velocities; it is a second-order method that costs two evaluations per step. DPM-Solver++(2M) is also second order but reuses the evaluation of the previous step, so it costs one evaluation per step.',
      'On log axes, an error of order \\(h^p\\) at NFE proportional to 1/h is a line with slope −p. Compare the three lines, and switch to a grid that is uniform in the log-SNR \\(\\lambda_t\\) instead of in t. The right plot shows the end points at the chosen NFE and where each solver evaluates the field.',
    ],
    tex: `\\text{Heun:}\\quad \\tilde\\x = \\x_k + ${h}\\,\\vel_{t_k}(\\x_k),\\qquad \\x_{k+1} = \\x_k + \\tfrac{${h}}{2}\\big(\\vel_{t_k}(\\x_k) + \\vel_{t_{k+1}}(\\tilde\\x)\\big)`,
    after: ['On the uniform grid, W1 at NFE 2, 4, 8, 16, 32 and 64 is 0.628, 0.320, 0.165, 0.084, 0.042 and 0.021 for Euler, 0.912, 0.142, 0.022, 0.0059, 0.0017 and 0.0010 for Heun, and 0.628, 0.203, 0.034, 0.0079, 0.0027 and 0.0012 for DPM-Solver++(2M). The slopes between NFE 4 and 16 are 0.97, 2.30 and 2.34, close to the orders 1, 2 and 2. Beyond NFE 32 the second-order solvers reach the floor of 0.0009 set by the finite sample. At NFE 2 Heun has a single step and is worse than Euler, so order only pays once the steps are small. At NFE 16 Heun is 14 times better than Euler for the same compute, which is a larger gain than any of the sampling methods below gives for a factor of 14 in NFE.', 'The log-SNR grid is worse here for every solver at every NFE above 2; at NFE 8, for example, Euler gives 0.256 instead of 0.165 and Heun 0.080 instead of 0.022. The choice of grid depends on the data and the path, and a grid tuned for images is not automatically right for another problem.'],
    check: {
      q: 'Between NFE 4 and 16, Euler drops from 0.320 to 0.084 and Heun from 0.142 to 0.0059. What orders do these numbers give?',
      opts: [
        ['About 1 for Euler and about 2 for Heun: the order is log₂ of the ratio of the errors divided by log₂ of the ratio of the NFE, which is 2 here.', true],
        ['About 2 for Euler and about 4 for Heun, because the NFE grows by a factor of 4.', false],
        ['The order cannot be read off, because Heun spends two evaluations per step.', false],
      ],
      fb: { true: 'Right: log₂(0.320/0.084)/2 ≈ 0.97 and log₂(0.142/0.0059)/2 ≈ 2.3. The extra evaluation of Heun changes the constant, not the slope, because NFE is proportional to the number of steps for every solver.', false: 'The slope of a log-log plot is the change in log error divided by the change in log NFE. The NFE grows by a factor of 4, which is two doublings.' },
    },
  },
  {
    id: 'bestofn', kicker: 'Many samples, one winner', title: 'Best-of-N', scene: 'bestofn',
    prose: [
      'The simplest way to spend more compute is to draw N samples and keep the one with the highest reward. Here the reward is a narrow bump between the left and the middle mode, where the data have little mass, so under \\(\\pdata\\) the average reward is only 0.044. The winner has a density that can be written down exactly.',
      'A principled target for the same trade-off is the tilted distribution \\(p^*\\) of §4, which maximizes the reward minus β times the KL to the data. For each N the plot shows \\(p^*\\) at the β that has the same KL as best-of-N, so the two use the same budget of deviation from the model. Move N and compare where the two put their mass.',
    ],
    tex: `${pN}(${x}) = ${N}\\,${pd}(${x})\\,${Fr}\\big(${r}(${x})\\big)^{${N} - 1},\\qquad \\KL(${pN}\\,\\|\\,${pd}) = \\log ${N} - \\tfrac{${N} - 1}{${N}}`,
    deriv: {
      have: [`N independent samples from \\(\\pdata\\), and \\(F_r(v) = P(r(X) \\le v)\\) for \\(X \\sim \\pdata\\), continuous so that there are no ties.`],
      goal: `p_N(\\x) = N\\,\\pdata(\\x)\\,F_r(r(\\x))^{N - 1}`,
      plan: 'The winner lies in a small interval around x when one of the N samples does and the other N − 1 have a smaller reward.',
      title: 'The density of the best of N', lines: [
        { tex: `P(\\text{winner} \\in \\dd\\x) = \\sum_{i=1}^{N} P\\big(X_i \\in \\dd\\x,\\ r(X_j) < r(\\x)\\ \\forall j \\ne i\\big)`, why: 'The events for different i are disjoint because there are no ties.', uses: ['expectation'] },
        { tex: `= ${CHG('N')}\\,\\pdata(\\x)\\,\\dd\\x\\ ${CHG('\\prod_{j \\ne i} P\\big(r(X_j) < r(\\x)\\big)')}`, why: 'The samples are independent and exchangeable, so each of the N terms is the same product.', uses: [] },
        { tex: `= N\\,\\pdata(\\x)\\,${CHG('F_r(r(\\x))^{N - 1}')}\\,\\dd\\x`, why: 'Each of the N − 1 factors is the reward CDF at r(x).', uses: [] },
        { tex: `\\KL(p_N\\,\\|\\,\\pdata) = \\E_{p_N}\\big[\\log N + (N - 1)\\log ${CHG('U')}\\big],\\quad U = F_r(r(\\x))`, why: 'The log of the density ratio. Under \\(\\pdata\\) the variable U is uniform on [0, 1], and the winner has the largest of N uniforms.', uses: ['kl', 'log-rules'] },
        { tex: `= \\log N + (N - 1)\\int_0^1 N u^{N - 1}\\log u\\,\\dd u = ${CHG('\\log N - \\tfrac{N - 1}{N}')}`, why: 'The maximum of N uniforms has density \\(N u^{N-1}\\), and \\(\\int_0^1 N u^{N-1}\\log u\\,\\dd u = -1/N\\).', uses: ['expectation', 'ibp'] },
      ],
    },
    after: ['The average reward of the winner at N = 2, 4, 8, 16, 32, 64 and 128 is 0.085, 0.158, 0.277, 0.447, 0.641, 0.813 and 0.924. At N = 8 the KL is 1.204, the matched temperature is β = 0.234 and \\(p^*\\) has an average reward of 0.475, against 0.277 for best-of-N at the same KL. The KL of best-of-N grows only like log N, but the winner moves to the top of the bump: the W1 distance between \\(p_N\\) and the matched \\(p^*\\) is 0.416 at N = 8 and falls to 0.013 at N = 128 only because both have then collapsed onto the bump (β = 0.105).', 'Best-of-N with a fixed N is therefore a poor use of a KL budget at small N, where it keeps most of its mass in the data modes, and it does not sample \\(p^*\\) at any N. Steps 3, 4 and 6 look for methods that do better at the same NFE.'],
    check: {
      q: 'Why is the KL of best-of-N the same, log N − (N − 1)/N, for every reward and every data distribution?',
      opts: [
        ['The density ratio depends on x only through the rank \\(F_r(r(\\x))\\), and that rank is uniform under \\(\\pdata\\) whatever r and \\(\\pdata\\) are, as long as there are no ties.', true],
        ['Because the reward is bounded by 1 here.', false],
        ['It is not the same: the formula is an upper bound that holds with equality only for this reward.', false],
      ],
      fb: { true: 'Right. This is why the KL alone does not describe what best-of-N does. Two rewards with the same N cost the same KL but can move the samples to very different places.', false: 'Look at the density ratio \\(p_N/\\pdata\\). Which function of x does it depend on, and what is the law of that function under \\(\\pdata\\)?' },
    },
  },
  {
    id: 'search', kicker: 'Optimizing the input', title: 'Searching the noise', scene: 'search',
    prose: [
      'A deterministic sampler is a function: the flow map \\(X_{0\\to 1}\\) takes a noise to a sample. Searching over the noise is therefore an optimization problem in which only the reward of the end point is observed. Random search draws n noises and keeps the best end point, which is exactly best-of-N. Local search starts from the best point so far and proposes Gaussian steps of size ρ, keeping a step when the reward improves. Trust-region noise search (<a href="https://arxiv.org/abs/2603.14504">Schweiger, Cremers and Ram, 2026</a>) spends a fifth of the budget on random draws and then keeps a few boxes around the best noises found so far. It proposes points inside each box, enlarges a box after repeated improvements and shrinks it after repeated failures, so that it explores several regions at once and refines each of them.',
      `There are two rewards. The first is \\(-|\\x - \\bm a|^2/2\\) for a point a beyond the right mode, so the best samples lie outside the data. The second is a narrow bump of width 0.15 at a point b on the edge of the right mode; about one data sample in 590 reaches half its height. The left plane is the noise, shaded by the reward of its end point, with circles at the typical norm \\(\\sqrt 2\\) and at 3. Each evaluation is one ODE solve with RK4 at 40 NFE. Switch the reward, and move the budget n and the step size ρ of local search.`,
    ],
    tex: `${x}_1 = ${X}(${z}),\\qquad \\max_{${z}}\\ ${r}\\big(${X}(${z})\\big)`,
    after: ['For the reward beyond the data, with n = 40 evaluations (1600 NFE) and ρ = 0.4, random search reaches r = −0.838 with a noise of norm 2.27, and its end point (1.85, 0.33) has data density 7.3 · 10⁻³. Local search reaches r = −0.011, much closer to a, but its noise has norm 4.02: a Gaussian in 2D puts only 3.1 · 10⁻⁴ of its mass beyond that norm, and the end point (2.25, 1.51) has data density 2.2 · 10⁻⁵. Trust-region search is still near the typical set at this budget (r = −0.885, norm 1.78). With n = 160, random search barely improves (−0.779), trust-region search reaches −0.062 with a noise of norm 3.63, and local search reaches −0.001 with norm 4.30. Averaged over 50 seeds at n = 160, the best rewards are −0.783, −0.055 and −0.006. On a smooth reward with one optimum, a single local search with a good step size is hard to beat, and every search that succeeds leaves the data, because that is where the reward is high.', 'The rare reward inside the data is a different problem. With n = 160 random search never comes near the bump in this run (best r = 0.005), while local search reaches 0.959 and trust-region search 0.991, with noises of norm 2.47 and 2.52 and end points at a data density of about 0.02, inside the data. Averaged over 100 seeds the best rewards at n = 40, 80 and 160 are 0.065, 0.120 and 0.262 for random search, 0.796, 0.941 and 0.979 for local search, and 0.313, 0.897 and 0.996 for trust-region search. The boxes need a few rounds before they pay off, and then they find the narrow peak more reliably than a single chain. Note that the norm of the noise alone does not keep a search plausible: the boxes do not constrain it, and what keeps the samples in the data here is that the reward itself lies in the data.'],
    check: {
      q: 'Local search reaches a much higher reward than random search on the reward beyond the data at the same NFE. What is the price?',
      opts: [
        ['Its best noise has a norm far above the typical √2, where the Gaussian has little mass, so its end point lies where the data density is tiny. The reward rose because the sample left the data.', true],
        ['Nothing: both searches return a sample of the model, so both are equally plausible under the data.', false],
        ['Local search needs gradients of the flow map, which random search does not.', false],
      ],
      fb: { true: 'Right. Every noise gives a valid output of the ODE, but not every noise is a plausible input. On the rare reward inside the data the same searches stay at a typical norm, so the price comes from where the reward is high, not from searching as such.', false: 'Read off the norm of the best noise and the data density at its end point in the readout. Where does a standard Gaussian in 2D put its mass?' },
    },
  },
  {
    id: 'flowgrad', kicker: 'Gradients through the sampler', title: 'Optimizing through the flow', scene: 'flowgrad',
    prose: [
      'When the reward is differentiable, the search can follow its gradient instead of guessing. D-Flow (<a href="https://arxiv.org/abs/2402.14017">Ben-Hamu et al., 2024</a>) differentiates the reward of the end point through the whole ODE solve and takes gradient steps on the noise. OC-Flow (<a href="https://arxiv.org/abs/2410.18070">Wang et al., 2024</a>) keeps the noise and adds a control \\(\\bm\\theta_k\\) to the velocity at every step, which is an optimal control problem with the reward at the end and a cost on the energy of the control. Both need one forward solve and one backward pass per iteration, counted here as 80 NFE.',
      'The backward pass is the same for both. A costate μ starts at the gradient of the reward at the end point and is carried back through the transposed Jacobians of the steps. Its value at a step is the gradient of the reward with respect to the point at that step, so OC-Flow uses it at every step and D-Flow uses it only at the start. The penalty β weighs the reward against the norm of the noise for D-Flow and against the control energy for OC-Flow. Switch the reward, and move the number of iterations and β.',
    ],
    tex: `\\max_{${S('th')}}\\ ${r}(\\x_K) - \\frac{${beta}}{2}\\sum_k h\\,|${S('th')}_k|^2,\\qquad \\x_{k+1} = \\x_k + h\\big(\\vel_{t_k}(\\x_k) + ${S('th')}_k\\big)`,
    deriv: {
      have: ['The controlled Euler scheme \\(\\x_{k+1} = \\x_k + h(\\vel_{t_k}(\\x_k) + \\bm\\theta_k)\\) for k = 0, …, K − 1, from a fixed noise \\(\\x_0\\).', 'The objective \\(J = r(\\x_K) - \\frac{\\beta}{2}\\sum_k h\\,|\\bm\\theta_k|^2\\).'],
      goal: `\\nabla_{\\bm\\theta_k} J = h\\,(\\bm\\mu_{k+1} - \\beta\\,\\bm\\theta_k),\\quad \\bm\\mu_k = (I + h\\,\\nabla\\vel_{t_k})^\\top\\bm\\mu_{k+1},\\ \\bm\\mu_K = \\nabla r(\\x_K)`,
      plan: 'A control acts on the path only through the next point. Define the costate as the gradient of the reward with respect to a point of the path, and get it backward by the chain rule.',
      title: 'The gradient of the end reward', lines: [
        { tex: `\\nabla_{\\bm\\theta_k} J = \\Big(\\frac{\\partial \\x_{k+1}}{\\partial \\bm\\theta_k}\\Big)^{\\!\\top} \\nabla_{\\x_{k+1}} r(\\x_K) - \\beta h\\,\\bm\\theta_k`, why: 'The control at step k changes \\(\\x_{k+1}\\) and, through it, every later point; the penalty depends on \\(\\bm\\theta_k\\) directly.', uses: ['chain-rule'] },
        { tex: `= ${CHG('h')}\\,${CHG('\\bm\\mu_{k+1}')} - \\beta h\\,\\bm\\theta_k,\\qquad \\bm\\mu_k := \\nabla_{\\x_k} r(\\x_K)`, why: 'The control enters the step additively with factor h, and the costate is defined as the gradient of the end reward with respect to the point.', uses: [] },
        { tex: `\\bm\\mu_k = \\Big(\\frac{\\partial \\x_{k+1}}{\\partial \\x_k}\\Big)^{\\!\\top}\\bm\\mu_{k+1} = ${CHG('(I + h\\,\\nabla\\vel_{t_k}(\\x_k))^\\top')}\\,\\bm\\mu_{k+1}`, why: 'The chain rule through one Euler step. This is the backward pass, and it needs one vector-Jacobian product of the field per step.', uses: ['chain-rule'] },
        { tex: `\\nabla_{\\x_0}\\, r(\\x_K) = ${CHG('\\bm\\mu_0')}`, why: 'With all controls at zero, the same recursion run to k = 0 gives the gradient with respect to the noise. D-Flow ascends this gradient, so it is OC-Flow with a single control, placed at the start.', uses: [] },
        { tex: `\\bm\\theta_k \\leftarrow \\bm\\theta_k + \\eta\\,(\\bm\\mu_{k+1} - \\beta\\,\\bm\\theta_k)`, why: 'A gradient step of size η/h on each control. At a fixed point \\(\\bm\\theta_k = \\bm\\mu_{k+1}/\\beta\\): the optimal control is the costate scaled by 1/β.', uses: [] },
      ],
    },
    after: ['For the reward beyond the data with β = 0, both methods start from the same noise, whose end point (−0.34, −0.95) lies in the left mode with r = −6.75. After 20 iterations (1600 NFE) D-Flow reaches r = −0.400 with a noise of norm 2.57 and data density 2.4 · 10⁻³ at the end point; after 80 iterations it reaches −0.020 with norm 3.94 and density 1.7 · 10⁻⁵. Local search of step 3 reached −0.011 with the same 1600 NFE, so the gradient does not pay for its backward pass on this smooth 2D problem. OC-Flow lands exactly on a after 20 iterations (r = −1.9 · 10⁻⁸) with a control energy of 17.96: with a control at every step and no penalty, any point can be reached, however far it lies from the data.', 'β trades reward for plausibility. With β = 0.3 after 80 iterations, D-Flow reaches r = −0.662 with norm 2.11, and OC-Flow −0.406 with energy 2.75 at data density 7.4 · 10⁻⁴; with β = 1, D-Flow reaches −1.488 with norm 1.14 and OC-Flow −1.860 with energy 1.20, both at a density of a few 10⁻². On the rare reward inside the data neither method moves: the end point lies so far from the bump that the gradient there is below 10⁻⁶⁹, and with β > 0 D-Flow only pulls the noise towards the origin. Step 3 found the same bump with 160 black-box evaluations.'],
    check: {
      q: 'On the rare reward both gradient methods stay where they start, while trust-region search finds the bump. Why?',
      opts: [
        ['The bump is narrow and far from the first end point, so its gradient there is numerically zero and carries no direction. A search that only compares reward values still finds the bump once one of its proposals lands near it.', true],
        ['The gradient through the ODE is wrong for narrow rewards, because RK4 with 10 steps is too coarse.', false],
        ['Gradient methods cannot be used on rewards that lie inside the data.', false],
      ],
      fb: { true: 'Right. Gradients are local information. They are very efficient once the iterate is in the basin of the reward, which is why practical methods often combine a black-box or multi-start search with gradient refinement, or smooth the reward first.', false: 'The first end point is at a distance of about 2.7 from the bump of width 0.15. How large is \\(e^{-d^2/(2 \\cdot 0.15^2)}\\) there?' },
    },
  },
  {
    id: 'dps', kicker: 'Conditioning on a measurement', title: 'Inverse problems', scene: 'dps',
    prose: [
      `In an inverse problem the target is not a reward but a posterior. Here the first coordinate of a sample is measured with Gaussian noise, \\(y = x^{(1)} + \\N(0, \\sigma_y^2)\\), and the sampler should draw from p(x | y). The posterior field adds the gradient of \\(\\log p_t(y \\mid \\x_t)\\) to the score, but this likelihood of the noisy point is an integral over the unknown clean point.`,
      'DPS replaces the clean point by its posterior mean \\(\\hx_1\\), which the denoiser provides. ΠGDM keeps a Gaussian around \\(\\hx_1\\) with the covariance of the clean point given the noisy one, which widens the likelihood when that covariance is large. Here both use the exact mean and covariance of the mixture, so the only error is the approximation itself. Compare both with the exact posterior field, and change y, \\(\\sigma_y\\) and the DPS scale ζ.',
    ],
    tex: `\\nabla\\log p_t(${y}\\mid\\x_t) \\approx ${zeta}\\,\\nabla_{\\x_t}\\log \\N\\big(${y};\\ \\hat x^{(1)}_1(\\x_t),\\ ${sy}^2\\big)`,
    deriv: {
      have: ['The likelihood \\(p(y \\mid \\x_1) = \\N(y;\\ x_1^{(1)}, \\sigma_y^2)\\) of the clean point.', 'The denoiser \\(\\hx_1(\\x_t) = \\E[\\x_1 \\mid \\x_t]\\) and the posterior covariance \\(\\Sigma_t(\\x_t) = \\mathrm{Cov}[\\x_1 \\mid \\x_t]\\).'],
      goal: `p_t(y \\mid \\x_t) \\approx \\N\\big(y;\\ \\hat x_1^{(1)},\\ \\sigma_y^2 + [\\Sigma_t]_{11}\\big)`,
      plan: 'Write the likelihood of the noisy point as an average over the clean point, then replace the law of the clean point by a simpler one.',
      title: 'The likelihood of a noisy point', lines: [
        { tex: `p_t(y \\mid \\x_t) = \\int p(y \\mid \\x_1)\\,p(\\x_1 \\mid \\x_t)\\,\\dd\\x_1`, why: 'y depends on \\(\\x_t\\) only through \\(\\x_1\\), so condition on \\(\\x_1\\) and average.', uses: ['cond-exp', 'bayes'] },
        { tex: `\\approx \\int p(y \\mid \\x_1)\\,${CHG('\\delta_{\\hx_1(\\x_t)}(\\x_1)')}\\,\\dd\\x_1 = ${CHG('\\N(y;\\ \\hat x_1^{(1)},\\ \\sigma_y^2)')}`, why: 'DPS: put all the mass of \\(p(\\x_1 \\mid \\x_t)\\) at its mean.', uses: ['tweedie'] },
        { tex: `\\approx \\int p(y \\mid \\x_1)\\,${CHG('\\N(\\x_1;\\ \\hx_1,\\ \\Sigma_t)')}\\,\\dd\\x_1 = \\N(y;\\ \\hat x_1^{(1)},\\ ${CHG('\\sigma_y^2 + [\\Sigma_t]_{11}')})`, why: 'ΠGDM: a Gaussian with the right mean and covariance. A Gaussian measurement of a Gaussian variable is Gaussian with the variances added.', uses: ['gauss'] },
        { tex: `\\nabla_{\\x_t}\\log p_t(y \\mid \\x_t) \\approx \\frac{y - \\hat x_1^{(1)}}{s^2}\\,${CHG('\\nabla_{\\x_t}\\hat x_1^{(1)}')},\\quad \\nabla_{\\x_t}\\hx_1 = \\frac{t\\,\\Sigma_t}{(1 - t)^2}`, why: 'The chain rule, with \\(s^2\\) the variance of either approximation (ΠGDM treats it as fixed). The Jacobian of the denoiser is the posterior covariance, scaled, by differentiating Tweedie\u2019s formula.', uses: ['chain-rule', 'tweedie'] },
        { tex: `\\Delta\\vel_t = \\frac{1 - t}{t}\\,\\nabla\\log p_t(y \\mid \\x_t) = \\frac{${CHG('\\Sigma_t\\,\\bm e_1')}\\,(y - \\hat x_1^{(1)})}{s^2\\,(1 - t)}`, why: 'The velocity of the linear path is \\((\\hx_1 - \\x)/(1 - t)\\), and adding a term to the score moves \\(\\hx_1\\) by \\((1 - t)^2/t\\) times it.', uses: ['substitution'] },
      ],
    },
    after: ['With y = 0 and \\(\\sigma_y = 0.1\\), the measurement falls in the gap between the two lower modes, and the posterior puts 0.040, 0.035 and 0.925 of its mass on the left, right and top mode. The exact posterior field gives shares 0.05, 0.04 and 0.91 and W1 of 0.008 and 0.066 in the two coordinates. ΠGDM gives 0.18, 0.30 and 0.52 and DPS 0.04, 0.44 and 0.51, both with a W1 above 1 in the second coordinate. The 0.44 of DPS that it assigns to the right mode in fact sits at (0.00, −0.97): these points match the measurement but lie off the data, because the gradient pulled them there at early times, when \\(\\hx_1\\) was still a blurred average.', 'At y = 0.8 the posterior is 0.00, 0.60 and 0.40, ΠGDM gives 0.00, 0.64 and 0.36 with W1 = 0.143 in the second coordinate, and DPS gives 0.10 and 0.90 on the right and top modes with W1 = 1.68. With \\(\\sigma_y = 1\\), W1 in the first coordinate grows with the DPS scale: 0.110 at ζ = 0.5, 0.244 at ζ = 1 and 0.648 at ζ = 2. The scale that DPS uses in practice is tuned, because the value the approximation prescribes is not the one that works.'],
    check: {
      q: 'When is the DPS approximation exact, and why does it fail at early times?',
      opts: [
        ['It is exact as t → 1, where \\(\\Sigma_t \\to 0\\) and \\(p(\\x_1 \\mid \\x_t)\\) concentrates at its mean. At early times \\(\\Sigma_t\\) is large, \\(\\hx_1\\) is an average of the modes, and the likelihood at that average is not the average likelihood.', true],
        ['It is exact as t → 0, because the noise then dominates the measurement.', false],
        ['It is exact whenever \\(\\sigma_y\\) is small, at every t.', false],
      ],
      fb: { true: 'Right. ΠGDM keeps the spread and is better calibrated in the example, but a single Gaussian still cannot describe a clean point that may lie in one of three modes. The exact posterior field has no such error.', false: 'The approximation replaces \\(p(\\x_1 \\mid \\x_t)\\) by a point mass. When is that law actually close to a point mass?' },
    },
  },
  {
    id: 'smc', kicker: 'Resampling along the way', title: 'SMC against best-of-N', scene: 'smc',
    prose: [
      'Best-of-N decides only at the end. Sequential Monte Carlo runs N particles of the stochastic sampler together and, after every step, multiplies each weight by a potential that measures how the reward of its denoised estimate changed. When the weights become uneven, measured by the effective sample size, the population is resampled: particles with low weight are dropped and particles with high weight are copied. Compute moves early to the paths that look promising.',
      'The potentials telescope, so the final weight of a particle is \\(e^{r(\\x_1)/\\beta}\\) up to a constant, and SMC samples the tilted target \\(p^*\\) as N grows. The plots compare SMC, best-of-N with the same number of samples, and one sample with reward-gradient guidance, at equal NFE. Change β and N.',
    ],
    tex: `${G}(\\x_k, \\x_{k+1}) = \\exp\\Big(\\big(${r}(${xh}(\\x_{k+1})) - ${r}(${xh}(\\x_k))\\big)/${beta}\\Big)`,
    deriv: {
      have: ['A path \\(\\x_0, \\x_1, \\dots, \\x_K\\) of the stochastic sampler, with \\(\\x_K\\) at t = 1 and the law of \\(\\x_K\\) equal to \\(\\pdata\\).', 'The denoiser satisfies \\(\\hx_1(\\x_K) = \\x_K\\) at t = 1, and \\(\\hx_1(\\x_0) = \\E[\\x_1]\\) at t = 0, which is one constant point.'],
      goal: `\\prod_k G_k \\propto e^{r(\\x_K)/\\beta}`,
      plan: 'Multiply the potentials and cancel the middle terms.',
      title: 'Why the potentials target p*', lines: [
        { tex: `\\prod_{k=0}^{K-1} G_k = \\exp\\Big(\\tfrac{1}{\\beta}\\sum_{k=0}^{K-1}\\big(r(\\hx_1(\\x_{k+1})) - r(\\hx_1(\\x_k))\\big)\\Big)`, why: 'The product of exponentials is the exponential of the sum.', uses: ['log-rules'] },
        { tex: `= \\exp\\Big(\\tfrac{1}{\\beta}\\big(${CHG('r(\\hx_1(\\x_K)) - r(\\hx_1(\\x_0))')}\\big)\\Big)`, why: 'The sum telescopes: every intermediate term appears once with each sign.', uses: [] },
        { tex: `= ${CHG('e^{r(\\x_K)/\\beta}')}\\,/\\,e^{r(\\E[\\x_1])/\\beta}`, why: 'At t = 1 the denoiser returns its input, and at t = 0 it returns the mean of the data, which does not depend on the path.', uses: ['tweedie'] },
        { tex: `\\Rightarrow\\ \\text{law of } \\x_K \\text{ under the weights} \\propto \\pdata(\\x_K)\\,e^{r(\\x_K)/\\beta} = ${CHG('p^*')}`, why: 'Weighting paths of the sampler by a function of the end point reweights the law of the end point, \\(\\pdata\\), by that function.', uses: ['expectation'] },
      ],
    },
    after: ['At β = 0.2 the target has an average reward of 0.607. With N = 1, 4, 16 and 64 particles (50 NFE each), SMC reaches a W1 to \\(p^*\\) of 0.849, 0.186, 0.074 and 0.055, and best-of-N with the same N reaches 0.891, 0.411, 0.320 and 0.380. The average rewards show why: SMC gives 0.455, 0.633 and 0.564 at N = 4, 16 and 64, near the target, while best-of-N gives 0.174, 0.478 and 0.863 and passes the target on its way to the maximum of r. One SMC run at N = 16 resamples 7 times. Reward-gradient guidance at 50 NFE has W1 = 0.749 and an average reward of 0.099, because the gradient of a narrow bump is almost zero far from it.', 'At β = 0.1, SMC reaches a W1 of 0.012 from N = 8, guidance stays at 1.387, and best-of-N reaches 0.037 only at N = 64, where the target is nearly the maximum itself. At β = 0.4 guidance is good (0.096), because the target is close to the data and the gradient is a small correction, while best-of-N rises from 0.276 to 0.993 as N grows and overshoots. These numbers come from 400 seeded runs per method and setting, so they carry Monte Carlo error of about 0.02.'],
    check: {
      q: 'As N grows at fixed β, what does each method approach?',
      opts: [
        ['SMC approaches \\(p^*\\), because its weights are \\(e^{r/\\beta}\\) and it is consistent as N → ∞. Best-of-N approaches the maximum of r, because the winner of more draws has a higher rank.', true],
        ['Both approach \\(p^*\\), but SMC gets there with fewer samples.', false],
        ['Both approach the maximum of r, because both select by the reward.', false],
      ],
      fb: { true: 'Right. This is the main difference between the two: SMC samples a distribution that has a temperature, best-of-N has no temperature and keeps sharpening. In the readout, the reward of best-of-N passes the target and keeps rising.', false: 'Look at the average reward of each method in the readout as N grows, and compare it with the reward under \\(p^*\\).' },
    },
  },
  {
    id: 'verifier', kicker: 'What the selection sees', title: 'An imperfect verifier', scene: 'verifier',
    prose: [
      'All the methods above trust the reward. In practice the selection is made by a verifier, a learned reward model or a classifier, that only approximates the quantity one cares about. Here the verifier \\(\\hat r\\) sees the true reward plus an error, the sample with the highest \\(\\hat r\\) among N is selected, and the plot reports the true, or gold, reward of the selected sample.',
      'Two kinds of error are compared. Independent noise of size σ is added to every score. A systematic error of height b adds a spurious bump at x = 3.1, in the tail of the right mode, where the true reward is zero. Move σ and b and watch the gold reward as N grows from 1 to 1024.',
    ],
    tex: `\\text{select } \\arg\\max_{i \\le ${N}}\\ ${rh}(\\x_i),\\qquad ${rh} = ${r} + b\\,\\phi(\\x) + \\sigma\\,\\varepsilon_i`,
    after: ['With a perfect verifier the gold reward at N = 16, 64, 256 and 1024 is 0.446, 0.812, 0.975 and 0.998. With independent noise σ = 0.3 it is 0.307, 0.608, 0.833 and 0.902: lower, but still rising. With σ = 1 it rises only slowly, from 0.105 to 0.260, because the noise now hides most of the signal. Independent noise slows the scaling but does not reverse it.', 'With a systematic error b = 1.5 the gold reward is 0.383, 0.513, 0.207 and 0.002. It peaks at N = 64 and then falls to nothing, because with enough draws one sample lands in the spurious bump, whose score of 1.5 beats any true reward. With b = 1 the spurious bump ties with the true one and the gold reward saturates at 0.80; with b = 1.2 it peaks at 0.587 at N = 64 and falls to 0.025 at N = 1024. The systematic curve is exact on a grid; the noisy curve is Monte Carlo over 2000 runs per N.'],
    check: {
      q: 'Why does the gold reward keep rising under independent noise but collapse under a systematic error?',
      opts: [
        ['Independent noise does not prefer any region, so with more draws the highest score still tends to come from a high true reward. A systematic error has its own maximum, and with enough draws best-of-N finds that maximum instead.', true],
        ['The systematic error is larger than the noise, so any verifier error of that size would give the same collapse.', false],
        ['Best-of-N overfits the verifier only when N exceeds the number of training samples of the verifier.', false],
      ],
      fb: { true: 'Right. More compute makes the selection more faithful to the verifier, which helps only as far as the verifier is faithful to the goal. This is reward hacking at test time, and it is the reason for early stopping in N, ensembles of verifiers, or a KL penalty as in step 6.', false: 'Compare σ = 1 with b = 1.5. The noise is the larger error, yet its curve does not fall. What is different about where the two errors point?' },
    },
    closing: `<h3>Where this leads</h3>
      <p>Each step spent compute in a different place. The table collects the measured numbers at the default settings.</p>
      <table class="tbl"><thead><tr><th>Step</th><th>Method</th><th>Compute</th><th>Result</th></tr></thead><tbody>
        <tr><td>1</td><td>Euler, Heun, DPM-Solver++(2M)</td><td>16 NFE</td><td>W1 0.084, 0.0059, 0.0079</td></tr>
        <tr><td>2</td><td>Best-of-8 against \\(p^*\\) at the same KL</td><td>8 samples</td><td>E[r] 0.277 against 0.475</td></tr>
        <tr><td>3</td><td>Random, trust-region and local search, rare reward</td><td>6400 NFE</td><td>r 0.005, 0.991 and 0.959, noise norm 1.66, 2.52 and 2.47</td></tr>
        <tr><td>4</td><td>D-Flow and OC-Flow, reward beyond the data, β = 0</td><td>1600 NFE</td><td>r −0.400 and −1.9 · 10⁻⁸, data density 2.4 · 10⁻³ and 6.2 · 10⁻⁶</td></tr>
        <tr><td>5</td><td>Exact field, ΠGDM, DPS</td><td>1000 steps</td><td>top-mode share 0.91, 0.52, 0.51 (posterior 0.925)</td></tr>
        <tr><td>6</td><td>SMC, best-of-N, guidance, β = 0.2</td><td>800 NFE</td><td>W1 to \\(p^*\\) 0.074, 0.320, 0.749</td></tr>
        <tr><td>7</td><td>Best-of-1024, verifier with σ = 0.3 and with b = 1.5</td><td>1024 samples</td><td>gold reward 0.902 and 0.002</td></tr>
      </tbody></table>
      <p>The tilted target and its KL come from <a href="../post-training/">Reward fine-tuning and preference alignment</a> (§4), which reaches \\(p^*\\) by training instead of by sampling. SMC and guidance at a single budget appear in <a href="../couplings/">Couplings, transport and control</a> (§2.6) and <a href="../guidance/">Conditioning and guidance</a> (§2.11). For language diffusion, test-time compute is spent on remasking and confidence thresholds, the parallel-decoding trade-off of <a href="../discrete/">Discrete and language diffusion</a> (§2.9). <a href="../distillation/">Distillation</a> (§5) asks the opposite question: how few evaluations a sample can take. Animations <a href="../../animations.html#a13">A13, best-of-N against SMC</a>, and <a href="../../animations.html#a14">A14, posterior sampling</a>, show steps 3, 5 and 6 as films.</p>`,
  },
];
