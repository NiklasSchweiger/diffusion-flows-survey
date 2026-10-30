// Content of the DDPM tutorial: one entry per step. The notation is that of Ho et al. (2020): steps
// t = 0..T with x_0 the data and epsilon the noise. The survey's time is written s on this page.
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const sab = (t = 't') => `\\sqrt{\\bar\\alpha_{${t}}}`, s1ab = (t = 't') => `\\sqrt{1-\\bar\\alpha_{${t}}}`;
const epsT = `${S('epsth')}(${S('xk')}, ${S('k')})`;

export const INTRO = `
  <div class="kicker">Survey §2.2–2.3 · Interactive tutorial</div>
  <h1>DDPM: a chain of small Gaussian steps</h1>
  <p>Denoising diffusion probabilistic models (Ho et al., 2020) reach diffusion from a different direction than score matching. They fix a Markov chain that adds a little noise at each of T steps, fit a Gaussian model to every reverse step and train it with a variational bound. The loss they end up with, a plain squared error on the noise, turns out to be denoising score matching with one particular weighting. This tutorial derives DDPM in its own terms and then shows exactly where it meets the score and the flow-matching path.</p>
  <p>A warning about notation. This page uses the notation of Ho et al. (2020), so that you can read it next to the paper. DDPM counts steps \\(t = 0, \\dots, T\\) from the data to the noise, writes the data point as \\(\\x_0\\) and the noise as \\(\\bm\\epsilon\\), and writes \\(\\alpha_t\\) for the signal factor \\(1-\\beta_t\\) of a single step. Flow matching and the rest of the survey use the same letters for different things: there \\(t\\) is a continuous time with \\(t = 0\\) noise and \\(t = 1\\) data, \\(\\x_0\\) is the noise, and \\(\\alpha_t\\) is the signal coefficient of \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\). To keep the two apart, this page writes the survey's time as \\(s\\), so its path reads \\(\\x_s = \\alpha_s\\x_1 + \\sigma_s\\x_0\\). A \\(t\\) on this page is always a DDPM step, and step 7 translates between the two. The data are the one-dimensional mixture of the earlier tutorials, so every density, kernel and bound on this page is exact.</p>
  <ol class="toc">
    <li><a href="#forward">The forward chain</a></li>
    <li><a href="#kernel">One reverse step is almost Gaussian</a></li>
    <li><a href="#elbo">A variational bound with one term per step</a></li>
    <li><a href="#eps">Predicting the noise</a></li>
    <li><a href="#score">The noise predictor is a score</a></li>
    <li><a href="#samplers">Two samplers for one network</a></li>
    <li><a href="#clocks">One path, two clocks</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'forward', kicker: 'The forward process', title: 'The forward chain', scene: 'forward',
    prose: [
      'DDPM starts from a fixed Markov chain. Each step shrinks the current point by \\(\\sqrt{1-\\beta_t}\\) and adds Gaussian noise of variance \\(\\beta_t\\). The shrinking keeps the variance near one, so after T steps the data are forgotten and \\(\\x_T\\) is close to a standard Gaussian. Nothing in this chain is learned.',
      'Two Gaussian steps in a row are again one Gaussian step, so the chain can be skipped: \\(\\x_t\\) can be drawn from \\(\\x_0\\) in one go. This is what makes training cheap. The plot shows the density \\(p_t\\) of every level as a heat map and a few paths of the chain. Change T, the number of steps. The tutorial keeps the same continuous schedule for every T, so fewer steps means larger \\(\\beta_t\\).',
    ],
    tex: `\\begin{aligned} q(${S('xk')}\\mid${S('xkm')}) &= \\N\\big(\\sqrt{1-${S('beta')}}\\,${S('xkm')},\\ ${S('beta')} I\\big) \\\\ q(${S('xk')}\\mid${S('x0')}) &= \\N\\big(\\sqrt{${S('abar')}}\\,${S('x0')},\\ (1-${S('abar')}) I\\big), \\qquad ${S('abar')} = \\prod_{i=1}^{${S('k')}}(1-\\beta_i) \\end{aligned}`,
    deriv: {
      have: ['One step of the chain, \\(\\x_t = \\sqrt{\\alpha_t}\\,\\x_{t-1} + \\sqrt{\\beta_t}\\,\\bm\\epsilon_t\\) with \\(\\alpha_t = 1-\\beta_t\\) and independent \\(\\bm\\epsilon_t\\sim\\N(0, I)\\).'],
      goal: 'step',
      plan: 'Unroll two steps, merge the two independent noise terms into one, and repeat down to the data point.',
      lines: [
        { tex: `${S('xk')} = \\sqrt{\\alpha_t}\\,${CHG(`\\big(\\sqrt{\\alpha_{t-1}}\\,\\x_{t-2} + \\sqrt{\\beta_{t-1}}\\,\\bm\\epsilon_{t-1}\\big)`)} + \\sqrt{\\beta_t}\\,\\bm\\epsilon_t`, why: 'Insert the step from level t − 2 to t − 1.' },
        { tex: `= \\sqrt{\\alpha_t\\alpha_{t-1}}\\,\\x_{t-2} + ${CHG(`\\sqrt{\\alpha_t(1-\\alpha_{t-1}) + 1-\\alpha_t}\\;\\bar{\\bm\\epsilon}`)}`, why: 'A sum of independent centred Gaussians is a centred Gaussian whose variance is the sum of the variances.', uses: ['gauss'] },
        { tex: `= \\sqrt{\\alpha_t\\alpha_{t-1}}\\,\\x_{t-2} + ${CHG('\\sqrt{1-\\alpha_t\\alpha_{t-1}}')}\\;\\bar{\\bm\\epsilon}`, why: 'Expand: α_t − α_t α_{t−1} + 1 − α_t = 1 − α_t α_{t−1}.' },
        { tex: `${S('xk')} = ${CHG(`\\sqrt{${S('abar')}}`)}\\,${S('x0')} + ${CHG(`\\sqrt{1-${S('abar')}}`)}\\,${S('eps')}`, why: 'Repeat down to level 0. The signal factors multiply to \\(\\bar\\alpha_t\\), and the noise variance is always one minus the signal variance.' },
      ],
    },
    after: ['The last line is the interpolation \\(\\x_s = \\alpha_s\\x_1 + \\sigma_s\\x_0\\) of the survey with \\(\\alpha_s = \\sqrt{\\bar\\alpha_t}\\) and \\(\\sigma_s = \\sqrt{1-\\bar\\alpha_t}\\), written with the opposite subscripts. Because \\(\\alpha_s^2 + \\sigma_s^2 = 1\\), DDPM is a variance-preserving path.'],
    check: {
      q: 'Why does each step multiply \\(\\x_{t-1}\\) by \\(\\sqrt{1-\\beta_t}\\) before adding noise?',
      opts: [['So that a point with variance 1 keeps variance 1', true], ['So that the chain stays near the data mean', false], ['So that the steps can be taken in any order', false]],
      fb: { true: 'Right. The variance after a step is (1 − β_t) Var + β_t, which equals 1 when Var = 1. Without the shrinking the variance would grow by β_t at every step, as in the variance-exploding processes of score matching.', false: 'Compute the variance after one step: (1 − β_t) Var(x_{t−1}) + β_t. It stays at 1 when it starts at 1. The factor exists to keep the scale fixed, which is why DDPM is called variance preserving.' },
    },
  },
  {
    id: 'kernel', kicker: 'The reverse step', title: 'One reverse step is almost Gaussian', scene: 'kernel',
    prose: [
      'To generate, the chain must run backwards. The true reverse step \\(q(\\x_{t-1}\\mid\\x_t)\\) depends on the unknown data distribution, so it is not available. If the data point \\(\\x_0\\) is also given, the reverse step becomes a Gaussian with known mean and variance, by Bayes\' rule applied to two Gaussians. The true reverse step is the average of these Gaussians over the data points that could have produced \\(\\x_t\\).',
      'The plot shows this average for one point \\(\\x_t\\), which you set with the slider. The thin curves are posteriors for individual data points, the filled curve is the exact reverse kernel, and the dashed curve is a single Gaussian with the same mean and variance \\(\\tilde\\beta_t\\). Make T small and move to the middle levels: the kernel becomes wide and skewed. Make T large and it becomes a Gaussian. This is why DDPM can model every step as a Gaussian, \\(p_\\theta(\\x_{t-1}\\mid\\x_t) = \\N(\\mu_\\theta, \\sigma_t^2 I)\\), as long as the steps are small.',
    ],
    tex: `\\begin{aligned} ${S('qpost')} &= \\N\\big(${S('mut')}(${S('xk')}, ${S('x0')}),\\ ${S('btil')} I\\big) \\\\ ${S('mut')} &= \\frac{${sab('t-1')}\\,${S('beta')}}{1-${S('abar')}}\\,${S('x0')} + \\frac{\\sqrt{${S('alpha')}}\\,(1-\\bar\\alpha_{t-1})}{1-${S('abar')}}\\,${S('xk')}, \\qquad ${S('btil')} = \\frac{1-\\bar\\alpha_{t-1}}{1-${S('abar')}}\\,${S('beta')} \\end{aligned}`,
    deriv: {
      have: ['The step \\(q(\\x_t\\mid\\x_{t-1}) = \\N(\\sqrt{\\alpha_t}\\x_{t-1}, \\beta_t I)\\) and the shortcut \\(q(\\x_{t-1}\\mid\\x_0) = \\N(\\sqrt{\\bar\\alpha_{t-1}}\\x_0, (1-\\bar\\alpha_{t-1}) I)\\) of step 1.'],
      goal: 'step',
      plan: 'Write Bayes\' rule for \\(\\x_{t-1}\\). Rewrite both factors as Gaussian bumps in \\(\\x_{t-1}\\), each with a centre and a variance. Their product is again a Gaussian: add the precisions and average the centres with the precisions as weights. Then simplify the coefficients.',
      lines: [
        { tex: `${S('qpost')} \\propto q(${S('xk')}\\mid${S('xkm')})\\,q(${S('xkm')}\\mid${S('x0')})`, why: 'Bayes\' rule for \\(\\x_{t-1}\\), with \\(\\x_0\\) given throughout. The chain is Markov, so \\(\\x_t\\) depends on \\(\\x_0\\) only through \\(\\x_{t-1}\\). The normalization \\(q(\\x_t\\mid\\x_0)\\) does not contain \\(\\x_{t-1}\\), so we can drop it and find the shape in \\(\\x_{t-1}\\).', uses: ['bayes'] },
        { tex: `\\frac{\\|${S('xk')} - \\sqrt{${S('alpha')}}\\,${S('xkm')}\\|^2}{${S('beta')}} = \\frac{${CHG(S('alpha'))}\\,\\|${S('xkm')} - ${CHG(`${S('xk')}/\\sqrt{${S('alpha')}}`)}\\|^2}{${S('beta')}}`, why: 'The first factor, the forward step, is a Gaussian in \\(\\x_t\\) with this squared distance in its exponent. To read it as a function of \\(\\x_{t-1}\\), pull \\(\\sqrt{\\alpha_t}\\) out of the norm. Its square \\(\\alpha_t\\) comes out in front.', uses: ['gauss'] },
        { tex: `q(${S('xk')}\\mid${S('xkm')}) \\propto \\exp\\Big(-\\frac{\\|${S('xkm')} - ${CHG(`${S('xk')}/\\sqrt{${S('alpha')}}`)}\\|^2}{2\\,${CHG(`${S('beta')}/${S('alpha')}`)}}\\Big)`, why: 'Move \\(\\alpha_t\\) into the denominator. As a function of \\(\\x_{t-1}\\), the first factor is a Gaussian bump with centre \\(\\x_t/\\sqrt{\\alpha_t}\\) and variance \\(\\beta_t/\\alpha_t\\). It is not normalized in \\(\\x_{t-1}\\), but constants do not matter here.', uses: ['gauss'] },
        { tex: `q(${S('xkm')}\\mid${S('x0')}) \\propto \\exp\\Big(-\\frac{\\|${S('xkm')} - ${CHG(`${sab('t-1')}\\,${S('x0')}`)}\\|^2}{2\\,${CHG('(1-\\bar\\alpha_{t-1})')}}\\Big)`, why: 'The second factor is the shortcut of step 1 at level \\(t-1\\). It is already a Gaussian in \\(\\x_{t-1}\\), with centre \\(\\sqrt{\\bar\\alpha_{t-1}}\\x_0\\) and variance \\(1-\\bar\\alpha_{t-1}\\).', uses: ['gauss'] },
        { tex: `\\frac{1}{${S('btil')}} = \\frac{1}{v_1} + \\frac{1}{v_2}, \\qquad ${S('mut')} = ${S('btil')}\\Big(\\frac{\\bm m_1}{v_1} + \\frac{\\bm m_2}{v_2}\\Big)`, why: 'The product of two Gaussian bumps in the same variable is a Gaussian bump: add the exponents and complete the square. Its precision is the sum of the precisions, and its mean is the precision-weighted average of the centres. Here \\(\\bm m_1 = \\x_t/\\sqrt{\\alpha_t}\\), \\(v_1 = \\beta_t/\\alpha_t\\), \\(\\bm m_2 = \\sqrt{\\bar\\alpha_{t-1}}\\x_0\\) and \\(v_2 = 1-\\bar\\alpha_{t-1}\\).', uses: ['gauss-product'] },
        { tex: `\\frac{1}{${S('btil')}} = \\frac{${S('alpha')}}{${S('beta')}} + \\frac{1}{1-\\bar\\alpha_{t-1}} = \\frac{${CHG(`${S('alpha')}(1-\\bar\\alpha_{t-1}) + ${S('beta')}`)}}{${S('beta')}\\,(1-\\bar\\alpha_{t-1})}`, why: 'Insert the two variances and bring the precisions to a common denominator.' },
        { tex: `${S('alpha')}(1-\\bar\\alpha_{t-1}) + ${S('beta')} = ${S('alpha')} - ${S('abar')} + 1 - ${S('alpha')} = ${CHG(`1-${S('abar')}`)} \\quad\\Longrightarrow\\quad ${S('btil')} = \\frac{1-\\bar\\alpha_{t-1}}{1-${S('abar')}}\\,${S('beta')}`, why: 'Simplify the numerator with \\(\\alpha_t\\bar\\alpha_{t-1} = \\bar\\alpha_t\\) and \\(\\beta_t = 1-\\alpha_t\\). Inverting gives the variance in the formula.' },
        { tex: `${S('mut')} = ${S('btil')}\\Big(${CHG(`\\frac{\\sqrt{${S('alpha')}}}{${S('beta')}}\\,${S('xk')} + \\frac{${sab('t-1')}}{1-\\bar\\alpha_{t-1}}\\,${S('x0')}`)}\\Big)`, why: 'Insert the centres and variances into the mean. The first weighted centre is \\(\\bm m_1/v_1 = (\\x_t/\\sqrt{\\alpha_t})(\\alpha_t/\\beta_t)\\), which equals \\(\\sqrt{\\alpha_t}\\,\\x_t/\\beta_t\\), and the second is \\(\\sqrt{\\bar\\alpha_{t-1}}\\x_0/(1-\\bar\\alpha_{t-1})\\).' },
        { tex: `${S('btil')}\\,\\frac{\\sqrt{${S('alpha')}}}{${S('beta')}} = ${CHG(`\\frac{\\sqrt{${S('alpha')}}\\,(1-\\bar\\alpha_{t-1})}{1-${S('abar')}}`)}, \\qquad ${S('btil')}\\,\\frac{${sab('t-1')}}{1-\\bar\\alpha_{t-1}} = ${CHG(`\\frac{${sab('t-1')}\\,${S('beta')}}{1-${S('abar')}}`)}`, why: 'Multiply out with the variance of the line before. In the first coefficient \\(\\beta_t\\) cancels, in the second \\(1-\\bar\\alpha_{t-1}\\) cancels. These are the two coefficients of the formula.' },
        { tex: `${S('k')} = 1:\\quad \\bar\\alpha_0 = 1 \\quad\\Longrightarrow\\quad ${S('btil')} = 0, \\quad ${S('mut')} = ${S('x0')}`, why: 'A check. At the last step, \\(\\x_{t-1}\\) is the data point itself, so given \\(\\x_0\\) the posterior must sit exactly at \\(\\x_0\\). It does: the variance vanishes, the \\(\\x_t\\) coefficient is zero and the \\(\\x_0\\) coefficient is \\(\\beta_1/(1-\\bar\\alpha_1) = 1\\).' },
        { tex: `q(${S('xkm')}\\mid${S('xk')}) = \\E\\big[\\,${S('qpost')}\\;\\big|\\;${S('xk')}\\big]`, why: 'The true reverse kernel averages the Gaussian posteriors over \\(\\x_0\\sim q(\\x_0\\mid\\x_t)\\). That average is the filled curve in the plot. It is a mixture, not a Gaussian.', uses: ['bayes', 'cond-exp'] },
      ],
    },
    after: ['The mean of the exact kernel is \\(\\tilde\\mu_t(\\x_t, \\E[\\x_0\\mid\\x_t])\\), because \\(\\tilde\\mu_t\\) is linear in \\(\\x_0\\). Its variance is \\(\\tilde\\beta_t\\) plus the spread of the thin curves, which is negligible when \\(\\beta_t\\) is small. Feller (1949) showed that the reverse of a diffusion with small steps has Gaussian steps, and Sohl-Dickstein et al. (2015) built the first diffusion model on this fact.'],
    check: {
      q: 'T grows and the schedule stays the same, so every \\(\\beta_t\\) shrinks. What happens to the KL between the exact kernel and the dashed Gaussian?',
      opts: [['It goes to zero', true], ['It stays about the same', false], ['It grows, because there are more steps', false]],
      fb: { true: 'Right. Move T from 5 to 1000 and watch the readout fall by orders of magnitude. The per-step error shrinks faster than the number of steps grows, which step 3 shows through the bound.', false: 'Watch the KL readout as you move T from 5 to 1000. Each step becomes so short that the data point barely matters for where x_{t−1} lands, and all the thin curves sit on top of each other.' },
    },
  },
  {
    id: 'elbo', kicker: 'The training objective', title: 'A variational bound with one term per step', scene: 'elbo',
    prose: [
      'The model chain starts from \\(\\N(0, I)\\) and applies T Gaussian steps. Its likelihood \\(p_\\theta(\\x_0)\\) is an integral over all paths and cannot be computed. As in a variational autoencoder, the forward chain serves as the encoder, and Jensen\'s inequality gives an upper bound \\(L\\) on \\(-\\log p_\\theta(\\x_0)\\). Rewriting the forward chain with the posteriors of step 2 splits \\(L\\) into one KL term per step.',
      'Each KL compares two Gaussians, so it reduces to a squared distance between their means plus a constant set by the variances. The plot takes the best possible mean at every level and shows the terms \\(L_{t-1}\\), times T, over the levels on a log scale. The readout adds them up. Switch the model variance between \\(\\tilde\\beta_t\\) and \\(\\beta_t\\) and change T. The right panel shows the total against T, next to the entropy of the data, which no model can beat.',
    ],
    tex: `\\begin{aligned} -\\log p_\\theta(${S('x0')}) \\le ${S('L')} &= \\E_q\\Big[${S('LK')} + \\sum_{t=2}^{${S('K')}} ${S('Lk')} + ${S('L0')}\\Big] \\\\ ${S('Lk')} &= \\KL\\big(${S('qpost')}\\,\\big\\|\\,${S('ptheta')}\\big) \\end{aligned}`,
    deriv: {
      have: ['The model chain \\(p_\\theta(\\x_{0:T}) = \\N(\\x_T; 0, I)\\prod_t p_\\theta(\\x_{t-1}\\mid\\x_t)\\) and the forward chain \\(q(\\x_{1:T}\\mid\\x_0)\\).', 'The Gaussian posteriors of step 2.'],
      goal: 'step',
      plan: 'Bound the log likelihood with Jensen\'s inequality, factor the forward chain backwards through its posteriors, and group the factors into KL divergences.',
      lines: [
        { tex: `-\\log p_\\theta(${S('x0')}) \\le \\E_q\\Big[-\\log\\frac{p_\\theta(\\x_{0:T})}{q(\\x_{1:T}\\mid${S('x0')})}\\Big]`, why: 'Write \\(p_\\theta(\\x_0)\\) as the expectation under q of the ratio inside, and move the logarithm inside the expectation.', uses: ['jensen'] },
        { tex: `q(\\x_{1:T}\\mid${S('x0')}) = ${CHG(`q(\\x_T\\mid${S('x0')})\\prod_{t=2}^{T} ${S('qpost')}`)}`, why: 'Factor the same joint law starting from level T. Given \\(\\x_0\\), the chain is still Markov in the reverse direction, and its reverse steps are the posteriors of step 2.', uses: ['bayes'] },
        { tex: `${S('L')} = \\E_q\\Big[${CHG(`\\underbrace{\\KL\\big(q(\\x_T\\mid${S('x0')})\\,\\|\\,\\N(0, I)\\big)}_{${S('LK')}} + \\sum_{t=2}^{T}${S('Lk')} \\underbrace{- \\log p_\\theta(${S('x0')}\\mid\\x_1)}_{${S('L0')}}`)}\\Big]`, why: 'Pair each forward factor with the model factor of the same step. Each pair, averaged over its variable, is a KL divergence.', uses: ['kl', 'expectation'] },
        { tex: `${S('Lk')} = \\E\\,\\frac{\\|${S('mut')} - ${S('mutheta')}\\|^2}{2${S('sig2')}} + ${CHG(`\\tfrac12\\Big(\\log\\frac{${S('sig2')}}{${S('btil')}} + \\frac{${S('btil')}}{${S('sig2')}} - 1\\Big)`)}`, why: 'The KL between two Gaussians splits into a mean part and a variance part. The variance part vanishes for \\(\\sigma_t^2 = \\tilde\\beta_t\\).', uses: ['gauss-kl'] },
        { tex: `\\min_{${S('mutheta')}}\\ \\E\\|${S('mut')} - ${S('mutheta')}\\|^2 = \\Big(\\frac{${sab('t-1')}\\,${S('beta')}}{1-${S('abar')}}\\Big)^2\\,\\E\\,\\mathrm{Var}(${S('x0')}\\mid${S('xk')})`, why: 'The best mean is \\(\\E[\\tilde\\mu_t\\mid\\x_t]\\). What is left is the variance of \\(\\tilde\\mu_t\\) given \\(\\x_t\\), and only its \\(\\x_0\\) part varies. This is what the plot evaluates.', uses: ['regression-lemma'] },
      ],
    },
    after: ['With the best mean and \\(\\sigma_t^2 = \\beta_t\\), the bound at T = 1000 is within 0.002 nats of the entropy. The choice \\(\\tilde\\beta_t\\) leaves a larger gap here: it is the right variance when the data point is known, and the data of this page are spread out. Ho et al. (2020) note that \\(\\tilde\\beta_t\\) is optimal for a single data point and \\(\\beta_t\\) for Gaussian data. The likelihood tutorial (<a href="../likelihood/#kl">§2.5, step 2</a>) writes the same term in the survey\'s notation as a weighted denoising error.'],
    check: {
      q: 'Every model step is Gaussian, yet the bound approaches the entropy of a three-mode mixture as T grows. Why?',
      opts: [['Each true reverse step becomes Gaussian as the steps get small', true], ['The last term L_0 absorbs the mismatch', false], ['The variance of the model is learned', false]],
      fb: { true: 'Right. The non-Gaussian shape of the data is built up from many small Gaussian steps, each of which is almost exact. Step 2 showed the per-step error shrinking.', false: 'L_0 and the variances are fixed here. The bound tightens because each reverse step becomes almost exactly Gaussian when β_t is small, as step 2 showed, and many small Gaussian steps can build up a multimodal distribution.' },
    },
  },
  {
    id: 'eps', kicker: 'Parametrization', title: 'Predicting the noise', scene: 'eps',
    prose: [
      'The mean \\(\\tilde\\mu_t\\) depends on the unknown \\(\\x_0\\). Writing \\(\\x_0\\) through the noise, \\(\\x_0 = (\\x_t - \\sqrt{1-\\bar\\alpha_t}\\,\\bm\\epsilon)/\\sqrt{\\bar\\alpha_t}\\), makes \\(\\tilde\\mu_t\\) a function of \\(\\x_t\\) and \\(\\bm\\epsilon\\). Ho et al. give the model mean the same form with a network \\(\\bm\\epsilon_\\theta\\) in place of \\(\\bm\\epsilon\\). Each ELBO term then becomes a squared error on the noise, multiplied by a weight \\(\\lambda_t\\).',
      'Their main empirical finding was that dropping the weights gives better samples. The plot shows the weights \\(\\lambda_t\\) of both variance choices, divided by their average over the levels, and the constant weight of \\(L_{\\mathrm{simple}}\\). Move the probe to read off how much more or less each level counts.',
    ],
    tex: `\\begin{aligned} ${S('mutheta')}(${S('xk')}, ${S('k')}) &= \\frac{1}{\\sqrt{${S('alpha')}}}\\Big(${S('xk')} - \\frac{${S('beta')}}{${s1ab()}}\\,${epsT}\\Big) \\\\ ${S('Lk')} &= ${S('lam')}\\,\\E\\big\\|${S('eps')} - ${epsT}\\big\\|^2 + C_t, \\qquad ${S('lam')} = \\frac{${S('beta')}^2}{2${S('sig2')}\\,${S('alpha')}\\,(1-${S('abar')})} \\\\ ${S('Ls')} &= \\E_{${S('k')},\\,${S('x0')},\\,${S('eps')}}\\big\\|${S('eps')} - ${S('epsth')}\\big(${sab()}\\,${S('x0')} + ${s1ab()}\\,${S('eps')},\\ ${S('k')}\\big)\\big\\|^2 \\end{aligned}`,
    deriv: {
      have: ['The posterior mean \\(\\tilde\\mu_t\\) of step 2 and the term \\(L_{t-1} = \\E\\|\\tilde\\mu_t - \\mu_\\theta\\|^2/2\\sigma_t^2 + C_t\\) of step 3.', 'The shortcut \\(\\x_t = \\sqrt{\\bar\\alpha_t}\\x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\bm\\epsilon\\).'],
      goal: 'step',
      plan: 'Eliminate \\(\\x_0\\) from \\(\\tilde\\mu_t\\), give \\(\\mu_\\theta\\) the same form, and subtract.',
      lines: [
        { tex: `${S('mut')} = \\frac{${sab('t-1')}\\,${S('beta')}}{1-${S('abar')}}\\,${CHG(`\\frac{${S('xk')} - ${s1ab()}\\,${S('eps')}}{${sab()}}`)} + \\frac{\\sqrt{${S('alpha')}}\\,(1-\\bar\\alpha_{t-1})}{1-${S('abar')}}\\,${S('xk')}`, why: 'Solve the shortcut for \\(\\x_0\\) and insert it.', uses: ['substitution'] },
        { tex: `${S('mut')} = ${CHG(`\\frac{1}{\\sqrt{${S('alpha')}}}\\Big(${S('xk')} - \\frac{${S('beta')}}{${s1ab()}}\\,${S('eps')}\\Big)`)}`, why: 'Use \\(\\bar\\alpha_t = \\alpha_t\\bar\\alpha_{t-1}\\). The coefficient of \\(\\x_t\\) is \\((\\beta_t + \\alpha_t - \\bar\\alpha_t)/(\\sqrt{\\alpha_t}(1-\\bar\\alpha_t)) = 1/\\sqrt{\\alpha_t}\\).' },
        { tex: `${S('mut')} - ${S('mutheta')} = ${CHG(`\\frac{${S('beta')}}{\\sqrt{${S('alpha')}}\\,${s1ab()}}\\,\\big(${epsT} - ${S('eps')}\\big)`)}`, why: 'Give \\(\\mu_\\theta\\) the same form with \\(\\bm\\epsilon_\\theta\\) in place of \\(\\bm\\epsilon\\) and subtract. The \\(\\x_t\\) terms cancel.' },
        { tex: `${S('Lk')} = ${CHG(`\\frac{${S('beta')}^2}{2${S('sig2')}\\,${S('alpha')}\\,(1-${S('abar')})}`)}\\,\\E\\big\\|${S('eps')} - ${epsT}\\big\\|^2 + C_t`, why: 'Square, divide by \\(2\\sigma_t^2\\) and take the expectation.', uses: ['expectation'] },
      ],
    },
    after: ['Both weights are largest at the first levels, where little noise has been added. \\(L_{\\mathrm{simple}}\\) therefore moves effort from the almost clean levels, which matter for the likelihood but hardly for how samples look, to the noisy levels, where the global shape of a sample is decided. Kingma et al. (2021) and the likelihood tutorial (<a href="../likelihood/#weights">§2.5, step 5</a>) treat all such losses as one integrand with different weights.'],
    check: {
      q: 'Compared with the ELBO with \\(\\sigma_t^2 = \\tilde\\beta_t\\), how does \\(L_{\\mathrm{simple}}\\) treat the levels with little noise (small t)?',
      opts: [['It gives them less weight', true], ['It gives them more weight', false], ['The same weight', false]],
      fb: { true: 'Right. The ELBO weight at the first levels is many times its average, and L_simple sets it to the average. Hover near t/T = 0 to read the ratio.', false: 'Look at the left end of the plot. The ELBO weights are far above 1 there, and L_simple flattens them to 1, so the low-noise levels count less.' },
    },
  },
  {
    id: 'score', kicker: 'Meeting score matching', title: 'The noise predictor is a score', scene: 'score',
    prose: [
      '\\(L_{\\mathrm{simple}}\\) is a regression: the input is \\(\\x_t\\), the label is the noise \\(\\bm\\epsilon\\) that produced it, and many labels share one input. The best predictor is therefore the conditional mean \\(\\E[\\bm\\epsilon\\mid\\x_t]\\). Every dot in the plot is one training pair. The solid curve is the exact conditional mean, and the squares are the averages of the dots in narrow bins.',
      'The conditional score of one noisy point is \\(-\\bm\\epsilon/\\sqrt{1-\\bar\\alpha_t}\\), and the marginal score is its average. So the conditional mean of the noise is the score, rescaled. Turn on the score: the thin curve is \\(-\\nabla\\log p_t\\), and the dashed curve is the same function multiplied by \\(\\sqrt{1-\\bar\\alpha_t}\\). The dashed curve lands on the solid one at every level, so the blue of the solid curve shows only through its gaps. DDPM and denoising score matching train the same function and differ only in the scale of the output and the weight of each level.',
    ],
    tex: `\\begin{aligned} ${S('epsstar')}(${S('xk')}) = \\E[${S('eps')}\\mid${S('xk')}] &= -${s1ab()}\\;${S('score')}(${S('xk')}) \\\\ \\big\\|${epsT} - ${S('eps')}\\big\\|^2 &= (1-${S('abar')})\\,\\Big\\|\\s_\\theta - \\Big(-\\frac{${S('eps')}}{${s1ab()}}\\Big)\\Big\\|^2, \\quad \\s_\\theta = -\\frac{${S('epsth')}}{${s1ab()}} \\end{aligned}`,
    deriv: {
      have: ['The loss \\(L_{\\mathrm{simple}}\\) of step 4.', 'The shortcut \\(q(\\x_t\\mid\\x_0) = \\N(\\sqrt{\\bar\\alpha_t}\\x_0, (1-\\bar\\alpha_t) I)\\) of step 1.'],
      goal: 'step',
      plan: 'Find the minimizer of the regression, compute the score of the shortcut Gaussian, and average it over the data points that are consistent with \\(\\x_t\\).',
      lines: [
        { tex: `\\arg\\min_{${S('epsth')}}\\ \\E\\|${S('eps')} - ${epsT}\\|^2 = ${CHG(`\\E[${S('eps')}\\mid${S('xk')}]`)}`, why: 'The regression lemma: squared error is minimized by the conditional mean of the label.', uses: ['regression-lemma'] },
        { tex: `\\nabla\\log q(${S('xk')}\\mid${S('x0')}) = -\\frac{${S('xk')} - ${sab()}\\,${S('x0')}}{1-${S('abar')}} = ${CHG(`-\\frac{${S('eps')}}{${s1ab()}}`)}`, why: 'The score of a Gaussian, evaluated at \\(\\x_t = \\sqrt{\\bar\\alpha_t}\\x_0 + \\sqrt{1-\\bar\\alpha_t}\\bm\\epsilon\\).', uses: ['gauss', 'cond-score'] },
        { tex: `${S('score')}(${S('xk')}) = \\E\\big[\\nabla\\log q(${S('xk')}\\mid${S('x0')})\\,\\big|\\,${S('xk')}\\big] = ${CHG(`-\\frac{\\E[${S('eps')}\\mid${S('xk')}]}{${s1ab()}}`)}`, why: 'The marginal score is the posterior average of the conditional scores.', uses: ['marginal-score'] },
        { tex: `${S('epsstar')}(${S('xk')}) = ${CHG(`-${s1ab()}\\;${S('score')}(${S('xk')})`)}`, why: 'Combine the first and the third line.' },
      ],
    },
    after: ['Song and Ermon (2019) trained score networks with denoising score matching on a geometric ladder of noise levels at about the same time. In the survey\'s notation the first line is \\(\\hx_0 = -\\sigma_s\\nabla\\log p_s\\), the identity of the score-matching tutorial (<a href="../score-matching/#marginal">§2.2, step 5</a>), and \\(L_{\\mathrm{simple}}\\) is denoising score matching with weight \\(\\sigma_s^2\\). What separates the two lines of work is the forward process (variance preserving here, variance exploding in NCSN), the loss weight, and the sampler, not what the network learns.'],
    check: {
      q: 'At a level with \\(\\bar\\alpha_t = 0.75\\), a noise predictor outputs \\(\\epsilon_\\theta = 0.5\\) at some point. What score does it imply there?',
      answer: -1, tol: 0.01,
      fb: { true: 'Right. √(1 − ᾱ_t) = 0.5, so the score is −0.5/0.5 = −1.', false: 'Use ∇log p_t = −ε_θ/√(1 − ᾱ_t). With ᾱ_t = 0.75 the square root is 0.5, and −0.5/0.5 = −1.' },
    },
  },
  {
    id: 'samplers', kicker: 'Sampling', title: 'Two samplers for one network', scene: 'samplers',
    prose: [
      'DDPM samples by running its model chain: draw \\(\\x_T\\sim\\N(0, I)\\) and repeatedly apply the Gaussian step with the trained mean and fresh noise \\(\\bm z\\). This is called ancestral sampling. Inserting \\(\\bm\\epsilon_\\theta = -\\sqrt{1-\\bar\\alpha_t}\\,\\nabla\\log p_t\\) shows that the step is \\(\\x_t + \\beta_t(\\tfrac12\\x_t + \\nabla\\log p_t)\\) plus noise, to first order in \\(\\beta_t\\). That is one Euler–Maruyama step of the reverse-time SDE of the variance-preserving process.',
      'Song, Meng and Ermon (2021) noticed that the training loss only involves the shortcuts \\(q(\\x_t\\mid\\x_0)\\), never the steps of the chain. Many other processes have the same shortcuts, and each gives a sampler for the same network. DDIM writes the step as a family with noise scale \\(\\eta\\): \\(\\eta = 1\\) is the ancestral sampler, and \\(\\eta = 0\\) is deterministic and discretizes the probability-flow ODE. Change T and the sampler. All runs start from the same 2000 noise points, and both use the exact \\(\\bm\\epsilon^*\\).',
    ],
    tex: `\\begin{aligned} \\x_{t-1} &= ${sab('t-1')}\\;${S('x0hat')} + \\sqrt{1-\\bar\\alpha_{t-1} - \\sigma_t^2}\\;${epsT} + \\sigma_t\\,${S('z')}, \\qquad \\sigma_t = ${S('eta')}\\sqrt{${S('btil')}} \\\\ ${S('x0hat')} &= \\frac{${S('xk')} - ${s1ab()}\\,${epsT}}{${sab()}} \\end{aligned}`,
    after: ['DDIM is markedly better at small T here, and both samplers converge as T grows. The ancestral sampler levels off near the dotted line and fluctuates around it: even 2000 exact draws from the data are that far from the data distribution in W1, and fresh noise at every step makes its samples random in the same way. The SDE tutorial (<a href="../sde/#ours">§2.3, step 6</a>) derives both continuous-time samplers, and §3 of the survey covers faster solvers.'],
    check: {
      q: 'In this plot, which sampler needs fewer steps to reach W1 below 0.05?',
      opts: [['DDIM, with η = 0', true], ['Ancestral, with η = 1', false], ['Both need the same number', false]],
      fb: { true: 'Right. DDIM reaches it at T = 50, the ancestral sampler only at T = 200. Deterministic samplers are usually preferred when steps are few, and noise helps mainly to correct errors of a learned model.', false: 'Read the two curves in the lower panel at W1 = 0.05. The DDIM curve crosses it at T = 50, the ancestral curve only at T = 200.' },
    },
  },
  {
    id: 'clocks', kicker: 'Meeting flow matching', title: 'One path, two clocks', scene: 'clocks',
    prose: [
      'Step 1 showed that \\(\\x_t\\) is the interpolation of the survey with \\(\\alpha_s = \\sqrt{\\bar\\alpha_t}\\) and \\(\\sigma_s = \\sqrt{1-\\bar\\alpha_t}\\). The linear path of flow matching uses coefficients that sum to one. Dividing \\(\\x_t\\) by \\(\\sqrt{\\bar\\alpha_t} + \\sqrt{1-\\bar\\alpha_t}\\) gives exactly that form, so a DDPM level is a point of the flow-matching path at a rescaled time and a rescaled position.',
      'Since the rescaling is invertible, it does not change what can be inferred about the noise. The DDPM noise predictor at \\(\\x_t\\) and the flow-matching noise predictor at \\(c_t\\x_t\\) and time \\(s\\) are the same number. The lower panel draws both, computed independently from the two formulas, and reports their largest difference. The upper panel shows how DDPM\'s levels map to flow-matching time.',
    ],
    tex: `\\begin{aligned} ${S('c')}\\,${S('xk')} &= ${S('srv')}\\;${S('x0')} + (1-${S('srv')})\\;${S('eps')}, \\qquad ${S('srv')} = \\frac{${sab()}}{${sab()} + ${s1ab()}}, \\quad ${S('c')} = \\frac{1}{${sab()} + ${s1ab()}} \\\\ ${S('epsstar')}_{\\mathrm{DDPM}}(${S('xk')}, ${S('k')}) &= ${S('epsstar')}_{\\mathrm{FM}}(${S('c')}\\,${S('xk')},\\ ${S('srv')}) \\end{aligned}`,
    deriv: {
      have: ['\\(\\x_t = \\sqrt{\\bar\\alpha_t}\\x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\bm\\epsilon\\) in DDPM letters.', 'The linear path \\(\\x_s = s\\,\\x_1 + (1-s)\\,\\x_0\\) of flow matching, with data \\(\\x_1\\) and noise \\(\\x_0\\).'],
      goal: 'step',
      plan: 'Rescale \\(\\x_t\\) so that its two coefficients sum to one, read off s, and use that conditioning on an invertible function of a variable is the same as conditioning on the variable.',
      lines: [
        { tex: `${CHG(`${S('c')}`)}\\,${S('xk')} = ${CHG(`${S('c')}${sab()}`)}\\,${S('x0')} + ${CHG(`${S('c')}${s1ab()}`)}\\,${S('eps')}`, why: 'Multiply by \\(c_t = 1/(\\sqrt{\\bar\\alpha_t} + \\sqrt{1-\\bar\\alpha_t})\\). The two coefficients now sum to one.' },
        { tex: `${S('c')}\\,${S('xk')} = ${CHG(`${S('srv')}`)}\\,${S('x0')} + ${CHG(`(1-${S('srv')})`)}\\,${S('eps')}`, why: 'Call the first coefficient s. This is the linear path, with DDPM\'s data point in the role of \\(\\x_1\\) and its noise in the role of \\(\\x_0\\).' },
        { tex: `\\E[${S('eps')}\\mid${S('xk')}] = \\E[${S('eps')}\\mid ${CHG(`${S('c')}\\,${S('xk')}`)}]`, why: 'Multiplying by a fixed positive number loses no information, so the conditional mean is the same.', uses: ['cond-exp'] },
      ],
    },
    after: ['A network trained with \\(L_{\\mathrm{simple}}\\) can therefore be turned into a flow-matching velocity model and the other way round, by rescaling the input and changing the time variable. What differs between the two families is the clock, the loss weight over that clock, and the output that the network is asked to produce, as the comparison below lists.'],
    check: {
      q: 'At a DDPM level with \\(\\bar\\alpha_t = 0.5\\), what is the flow-matching time s?',
      answer: 0.5, tol: 0.01,
      fb: { true: 'Right. Both square roots are √0.5, so s = 1/2. The level where signal and noise have equal variance is the midpoint of the linear path.', false: 'With ᾱ_t = 0.5 both √ᾱ_t and √(1 − ᾱ_t) equal √0.5, so s = √0.5/(2√0.5) = 0.5.' },
    },
    closing: `<h3>DDPM, score matching and flow matching side by side</h3>
      <div class="cmp-wrap"><table class="cmp">
        <thead><tr><th></th><th>DDPM</th><th>Score matching (NCSN, SDE)</th><th>Flow matching</th></tr></thead>
        <tbody>
          <tr><th>Clock</th><td>Levels \\(t = 0, \\dots, T\\), data at \\(t = 0\\)</td><td>Noise scales \\(\\sigma_1 > \\dots > \\sigma_L\\), or continuous time of an SDE</td><td>\\(s\\in[0, 1]\\) (written t in the survey), data at \\(s = 1\\)</td></tr>
          <tr><th>Noising</th><td>Markov chain with variances \\(\\beta_t\\), variance preserving</td><td>Added noise of scale \\(\\sigma\\), variance exploding (NCSN), or a forward SDE</td><td>Interpolation \\(s\\,\\x_1 + (1-s)\\,\\x_0\\)</td></tr>
          <tr><th>Network output</th><td>Noise \\(\\bm\\epsilon_\\theta\\)</td><td>Score \\(\\s_\\theta\\)</td><td>Velocity \\(\\vel_\\theta\\)</td></tr>
          <tr><th>Loss</th><td>\\(\\|\\bm\\epsilon - \\bm\\epsilon_\\theta\\|^2\\)</td><td>\\(\\lambda(\\sigma)\\,\\|\\s_\\theta + \\bm\\epsilon/\\sigma\\|^2\\)</td><td>\\(\\|\\x_1 - \\x_0 - \\vel_\\theta\\|^2\\)</td></tr>
          <tr><th>Derived from</th><td>A variational bound, with the weights dropped</td><td>The Fisher divergence, made tractable by adding noise</td><td>Regression on conditional velocities</td></tr>
          <tr><th>Optimum</th><td>\\(\\E[\\bm\\epsilon\\mid\\x_t]\\)</td><td>\\(\\nabla\\log p_\\sigma\\)</td><td>\\(\\E[\\x_1 - \\x_0\\mid\\x_s]\\)</td></tr>
          <tr><th>Samplers</th><td>Ancestral, DDIM</td><td>Annealed Langevin, reverse SDE, probability-flow ODE</td><td>ODE</td></tr>
        </tbody>
      </table></div>
      <p>At their optima the three networks determine each other: step 5 converts noise to score, and step 7 converts DDPM levels to flow-matching times. The score-matching tutorial (<a href="../score-matching/#targets">§2.2, step 7</a>) converts noise and data predictions to velocities.</p>
      <h3>Where this leads</h3>
      <p><a href="../sde/">§2.3 of the survey</a> lets the number of steps go to infinity, which turns the chain into an SDE and the two samplers of step 6 into the reverse SDE and the probability-flow ODE. <a href="../flow-matching/">§2.4</a> builds the linear path of step 7 directly, and <a href="../likelihood/">§2.5</a> returns to the bound of step 3 in continuous time.</p>`,
  },
];
