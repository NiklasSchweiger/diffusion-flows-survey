// Content of the §2.5 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex and
// the derivation in appendix app:elbo; S(id) tags a symbol from symbols.js, CHG marks the part of a
// derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x1 = S('x1'), x0 = S('x0'), xt = S('xt'), xs = S('xs'), q = S('q'), pth = S('ptheta'), N = S('Nl');
const hx1 = S('hx1'), hx0 = S('hx0'), lam = S('lam'), snr = S('snr'), ell = S('ell'), Linf = S('Linf');
const mu = S('mu'), vs = S('vs'), w = S('w'), pi = S('pi'), u = S('u'), div = S('div'), pt = S('pt');
const lmin = `${lam}_{\\min}`, lmax = `${lam}_{\\max}`, lp = S('lprime');
const xi = (i) => S('xt', `\\x_{t_{${i}}}`), lat = S('xt', '\\x_{0:N-1}');
const hu = S('u', '\\hat\\vel');

export const INTRO = `
  <div class="kicker">Survey §2.5 · Interactive tutorial</div>
  <h1>The likelihood view: ELBO, VDM and loss weighting</h1>
  <p>The earlier tutorials justified every denoising loss by regression: its minimizer is the right conditional expectation. They did not say how much each noise level should count, and every weighting has the same minimizer when the network is flexible enough. This tutorial derives the one weighting with a probabilistic meaning. It treats a diffusion model as a deep variational autoencoder, shows that each term of its bound is a weighted denoising error, takes the continuous-time limit, and then reads the losses used in practice as reweightings of the same integrand. It ends with the exact likelihood of the probability-flow ODE.</p>
  <p>The convention is that of the survey: \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), where \\(t = 0\\) is noise and \\(t = 1\\) is data, and \\(\\lambda_t = \\log(\\alpha_t^2/\\sigma_t^2)\\) is the log signal-to-noise ratio. The data are the 1D Gaussian mixture of the earlier tutorials. The denoiser is the exact posterior mean, the best any network can do, so every bound on this page is the best one its weighting allows. Error integrals are computed by quadrature, not by sampling, unless a plot says otherwise.</p>
  <ol class="toc">
    <li><a href="#vae">A diffusion model is a deep VAE</a></li>
    <li><a href="#kl">One KL term is a weighted denoising error</a></li>
    <li><a href="#vdm">Continuous time: the VDM objective</a></li>
    <li><a href="#schedule">The schedule changes only the variance</a></li>
    <li><a href="#weights">Every loss is a weighting of one integrand</a></li>
    <li><a href="#monotone">Decreasing weights are ELBOs of noisy data</a></li>
    <li><a href="#exact">Exact likelihood through the probability-flow ODE</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'vae', kicker: 'The variational bound', title: 'A diffusion model is a deep VAE', scene: 'vae',
    prose: [
      'Fix \\(N\\) noise levels \\(t_0 < t_1 < \\dots < t_{N-1}\\) and treat the noisy points \\(\\x_{t_0}, \\dots, \\x_{t_{N-1}}\\) as latent variables. The encoder is the corruption process \\(q(\\x_t\\mid\\x_1) = \\N(\\alpha_t\\x_1, \\sigma_t^2 I)\\). It is fixed and has no parameters. The decoder is the generative chain: it starts from \\(\\N(0, I)\\) and removes noise one level at a time. As for any VAE, the negative log-likelihood of a data point is bounded by the formula below: a prior term, one denoising term per step, and a reconstruction term.',
      'The plot shows the latent levels for the 1D mixture, equally spaced in log-SNR from −9 to 9: each row is the density of \\(\\x_{t_i}\\), from noise at the bottom to data at the top. The decoder here is the best one possible, because it uses the exact data prediction (step 2 explains how). The right panel shows how far the averaged bound lies above the entropy of the data, the smallest value it could reach. Move \\(N\\). With few levels each decoder step must jump across a large part of the path, and the bound is useless. With many levels it approaches the entropy.',
    ],
    tex: `\\begin{aligned} -\\log ${pth}(${x1}) \\le{}& \\underbrace{\\KL\\big(${q}(${xi(0)}\\mid${x1})\\,\\|\\,p(${xi(0)})\\big)}_{\\text{prior}} + \\underbrace{\\E_{${q}}\\big[-\\log ${pth}(${x1}\\mid${xi('N-1')})\\big]}_{\\text{reconstruction}} \\\\ &+ \\sum_{i=1}^{${N}-1}\\E_{${q}}\\underbrace{\\KL\\big(${q}(${xi('i')}\\mid${xi('i-1')},${x1})\\,\\|\\,${pth}(${xi('i')}\\mid${xi('i-1')})\\big)}_{\\text{denoising step } i} \\end{aligned}`,
    deriv: {
      have: ['An encoder \\(q(\\x_{0:N-1}\\mid\\x_1)\\), the law of the latents given the data. It is Markov from data to noise.', 'A decoder that factorizes from noise to data: \\(p(\\x_{t_0})\\prod_{i=1}^{N-1}p_\\theta(\\x_{t_i}\\mid\\x_{t_{i-1}})\\,p_\\theta(\\x_1\\mid\\x_{t_{N-1}})\\).'],
      goal: 'step',
      plan: 'Write the likelihood as an expectation under the encoder and apply Jensen’s inequality. Then turn the encoder around with Bayes’ rule so that it factorizes in the same order as the decoder, and match the factors.',
      title: 'Derivation of the bound', lines: [
      { tex: `\\log ${pth}(${x1}) = \\log \\E_{${q}}\\Big[\\frac{${pth}(${x1}, ${lat})}{${q}(${lat}\\mid${x1})}\\Big]`, why: 'Multiply and divide the joint density of the model by the encoder density, and read the integral over the latents as an expectation under the encoder.', uses: ['bayes', 'expectation'] },
      { tex: `\\log ${pth}(${x1}) \\;${CHG('\\ge')}\\; \\E_{${q}}\\Big[\\log\\frac{${pth}(${x1}, ${lat})}{${q}(${lat}\\mid${x1})}\\Big]`, why: 'Jensen’s inequality moves the logarithm inside the expectation. The gap is the KL divergence from the encoder to the true posterior of the latents.', uses: ['jensen'] },
      { tex: `${q}(${lat}\\mid${x1}) = ${CHG(`${q}(${xi(0)}\\mid${x1})\\prod_{i=1}^{${N}-1}${q}(${xi('i')}\\mid${xi('i-1')},${x1})`)}`, why: 'The encoder adds noise from data to noise, so given the data each latent depends only on its less noisy neighbour. Bayes’ rule, applied one link at a time, writes the same joint law in the generative direction, from the noisiest latent to the cleanest.', uses: ['bayes'] },
      { tex: `\\log\\frac{${pth}(${x1}, ${lat})}{${q}(${lat}\\mid${x1})} = \\log\\frac{p(${xi(0)})}{${q}(${xi(0)}\\mid${x1})} + \\sum_{i=1}^{${N}-1}\\log\\frac{${pth}(${xi('i')}\\mid${xi('i-1')})}{${q}(${xi('i')}\\mid${xi('i-1')},${x1})} + \\log ${pth}(${x1}\\mid${xi('N-1')})`, why: 'Both joints now factorize in the same order, so the log of their ratio is a sum of log ratios of matching factors.', uses: ['log-rules'] },
      { tex: `-\\log ${pth}(${x1}) \\le ${CHG(`\\KL\\big(${q}(${xi(0)}\\mid${x1})\\,\\|\\,p(${xi(0)})\\big) + \\sum_{i=1}^{${N}-1}\\E_{${q}}\\KL(\\cdots) + \\E_{${q}}\\big[-\\log ${pth}(${x1}\\mid${xi('N-1')})\\big]`)}`, why: 'Take the expectation of each term and change the sign. The expected log ratio of a density to another, under the first, is a KL divergence; for step i the inner expectation over \\(\\x_{t_i}\\) gives the KL and the outer one averages over \\(\\x_{t_{i-1}}\\).', uses: ['kl'] },
    ] },
    after: ['The prior term is almost zero when the noisiest level is almost pure noise, and the reconstruction term depends only on the cleanest level. All the learning happens in the denoising terms. The next step computes one of them.'],
    check: {
      q: 'At \\(N = 2\\) the bound exceeds the entropy by thousands of nats. Why?',
      opts: [['A single Gaussian step cannot turn almost pure noise into a three-mode distribution', true], ['The prior term is large because \\(\\lambda_{\\min}\\) is too high', false], ['The decoder is not the exact posterior mean', false]],
      fb: { true: 'Right. With two levels there is one denoising step, from log-SNR −9 to 9. Its decoder is a narrow Gaussian around one prediction, but given almost pure noise the clean point could be in any of the three modes. The readout shows that almost all of the bound is in that one term.', false: 'The readout shows the prior term at about \\(10^{-4}\\), and the decoder already uses the exact data prediction. The whole excess is in the single denoising term: a narrow Gaussian step must predict a point that could lie in any of the three modes.' },
    },
  },
  {
    id: 'kl', kicker: 'Inside one term', title: 'One KL term is a weighted denoising error', scene: 'kl',
    prose: [
      'Take two levels \\(s < t\\), so that \\(\\x_s\\) is the noisier one. Given the clean point \\(\\x_1\\), the encoder posterior \\(q(\\x_t\\mid\\x_s, \\x_1)\\) is Gaussian, with a mean \\(\\bm\\mu(\\x_s, \\x_1)\\) that is linear in \\(\\x_1\\) and a variance \\(\\varsigma^2\\) that does not depend on it. The decoder does not know \\(\\x_1\\), so it uses the same Gaussian with \\(\\x_1\\) replaced by a prediction \\(\\hx_1(\\x_s, s)\\). The KL divergence between two Gaussians with the same variance is the squared distance of their means, and the formula below follows.',
      'The upper panel shows the density of \\(\\x_s\\); move the cursor to place the probe \\(\\x_s\\), and click to pin it. The lower panel shows what each side of the KL looks like at that point. The thin Gaussians are the encoder posteriors for 16 draws of \\(\\x_1\\) from its exact posterior given \\(\\x_s\\). The bold Gaussian is the decoder. Its width is the same, and its centre is the average of theirs. Move \\(\\Delta\\lambda\\): a small step has narrow Gaussians that sit close together.',
    ],
    tex: `\\KL\\big(${q}(${xt}\\mid${xs},${x1})\\,\\|\\,${pth}(${xt}\\mid${xs})\\big) = \\tfrac12\\big(${snr}(t) - ${snr}(s)\\big)\\,\\big\\|${x1} - ${hx1}(${xs}, s)\\big\\|^2`,
    deriv: {
      have: ['The encoder \\(q(\\x_t\\mid\\x_1) = \\N(\\alpha_t\\x_1, \\sigma_t^2 I)\\) and its step from \\(t\\) to \\(s\\): \\(\\x_s = \\alpha_{s|t}\\x_t + \\sigma_{s|t}\\bm\\epsilon\\), with \\(\\alpha_{s|t} = \\alpha_s/\\alpha_t\\) and \\(\\sigma_{s|t}^2 = \\sigma_s^2 - \\alpha_{s|t}^2\\sigma_t^2\\).', 'A decoder \\(p_\\theta(\\x_t\\mid\\x_s) = \\N\\big(\\bm\\mu(\\x_s, \\hx_1), \\varsigma^2 I\\big)\\) that copies the encoder posterior with \\(\\hx_1\\) in place of \\(\\x_1\\).'],
      goal: 'step',
      plan: 'Compute the encoder posterior by Bayes’ rule, take the KL of two Gaussians with equal variance, and simplify the weight in front of the squared error.',
      title: 'Derivation of one denoising term', lines: [
      { tex: `${q}(${xt}\\mid${xs},${x1}) \\;\\propto\\; \\N\\big(${xs};\\,\\alpha_{s|t}${xt},\\,\\sigma_{s|t}^2\\big)\\;\\N\\big(${xt};\\,\\alpha_t${x1},\\,\\sigma_t^2\\big)`, why: 'Bayes’ rule: the posterior of \\(\\x_t\\) is its prior given \\(\\x_1\\) times the likelihood of the noisier observation \\(\\x_s\\). The encoder is Markov, so \\(\\x_s\\) depends on \\(\\x_1\\) only through \\(\\x_t\\).', uses: ['bayes'] },
      { tex: `= \\N\\big(${mu}, ${vs}\\big), \\qquad ${CHG(`${vs} = \\frac{\\sigma_{s|t}^2\\sigma_t^2}{\\sigma_s^2}, \\quad ${mu} = \\frac{\\alpha_{s|t}\\sigma_t^2}{\\sigma_s^2}\\,${xs} + \\frac{\\alpha_t\\sigma_{s|t}^2}{\\sigma_s^2}\\,${x1}`)}`, why: 'A product of Gaussians in \\(\\x_t\\) is Gaussian. The precisions add, \\(\\alpha_{s|t}^2/\\sigma_{s|t}^2 + 1/\\sigma_t^2 = \\sigma_s^2/(\\sigma_{s|t}^2\\sigma_t^2)\\), and the mean is the precision-weighted average of the two guesses for \\(\\x_t\\).', uses: ['gauss'] },
      { tex: `\\KL = \\frac{\\big\\|${mu}(${xs}, ${x1}) - ${mu}(${xs}, ${hx1})\\big\\|^2}{2${vs}} = ${CHG(`\\frac{1}{2${vs}}\\,\\frac{\\alpha_t^2\\sigma_{s|t}^4}{\\sigma_s^4}\\,\\big\\|${x1} - ${hx1}\\big\\|^2`)}`, why: 'Both Gaussians have the variance \\(\\varsigma^2\\), so the KL is the squared distance of the means over \\(2\\varsigma^2\\). Only the \\(\\x_1\\) coefficient of \\(\\bm\\mu\\) differs between the two.', uses: ['gauss-kl'] },
      { tex: `= \\frac{\\alpha_t^2\\sigma_{s|t}^2}{2\\sigma_s^2\\sigma_t^2}\\,\\big\\|${x1} - ${hx1}\\big\\|^2 = ${CHG(`\\tfrac12\\Big(\\frac{\\alpha_t^2}{\\sigma_t^2} - \\frac{\\alpha_s^2}{\\sigma_s^2}\\Big)`)}\\,\\big\\|${x1} - ${hx1}\\big\\|^2`, why: 'Insert \\(\\varsigma^2\\) and cancel. Then substitute \\(\\sigma_{s|t}^2 = \\sigma_s^2 - \\alpha_s^2\\sigma_t^2/\\alpha_t^2\\): the factor \\(\\alpha_t^2\\sigma_{s|t}^2/(\\sigma_s^2\\sigma_t^2)\\) splits into a difference of two signal-to-noise ratios.', uses: [] },
      { tex: `\\KL = \\tfrac12\\big(${snr}(t) - ${snr}(s)\\big)\\,\\big\\|${x1} - ${hx1}(${xs}, s)\\big\\|^2`, why: 'With \\(\\mathrm{SNR}(t) = \\alpha_t^2/\\sigma_t^2 = e^{\\lambda_t}\\). Averaged over \\(\\x_1\\) given \\(\\x_s\\), the squared error is smallest for the posterior mean, and its minimum is the posterior variance.', uses: ['regression-lemma'] },
    ] },
    after: ['Training DDPM by maximum likelihood is therefore data prediction, with each step weighted by the signal-to-noise ratio it adds. The squared error cannot be zero, because \\(\\x_1\\) is random given \\(\\x_s\\); the readout shows its minimum, the posterior variance, which is what the best decoder pays.'],
    check: {
      q: 'Make \\(\\Delta\\lambda\\) small and then halve it. What happens to the expected KL of the step?',
      opts: [['It about halves, because the weight \\(\\mathrm{SNR}(t) - \\mathrm{SNR}(s)\\) is about \\(e^{\\lambda_s}\\Delta\\lambda\\)', true], ['It stays the same, because the posterior variance does not change', false], ['It falls to about a quarter, because the KL is quadratic in the step', false]],
      fb: { true: 'Right. The posterior variance of \\(\\x_1\\) given \\(\\x_s\\) does not depend on \\(\\Delta\\lambda\\), and for small steps the weight is proportional to the step. Twice as many steps of half the size cost about the same in total, which is why the sum has a limit in step 3. For large steps the weight \\(e^{\\lambda_s}(e^{\\Delta\\lambda} - 1)\\) grows faster than the step: from 1 to 0.5 it falls to 0.38 of its value, not 0.5.', false: 'The error \\(\\mathrm{Var}(\\x_1\\mid\\x_s)\\) does not depend on the step size, only the weight does. For a small step \\(e^{\\lambda_t} - e^{\\lambda_s} \\approx e^{\\lambda_s}\\Delta\\lambda\\), so the KL is linear in \\(\\Delta\\lambda\\). Compare the readout at Δλ = 0.2 and 0.1.' },
    },
  },
  {
    id: 'vdm', kicker: 'The continuous-time limit', title: 'Continuous time: the VDM objective', scene: 'vdm',
    prose: [
      'Sum the terms of step 2 over all levels and refine the grid. The sum is a Riemann sum, and it tends to an integral over time. Changing the variable from \\(t\\) to \\(\\lambda\\) and writing the error in noise coordinates gives the objective of the variational diffusion model, the formula below: the noise-prediction error with uniform weight in log-SNR.',
      'The plot shows the integrand \\(\\ell(\\lambda)\\) for the 1D mixture, computed exactly, and the bars of the discrete bound with \\(N\\) levels. Each bar spans one step, and its area is that step’s KL term; bars taller than the plot are cut and marked. With few levels, the bars far overshoot the curve, because each step uses the error at its noisier end, where it is largest in data coordinates. As \\(N\\) grows, the bars fill the area under the curve and the bound tends to \\(\\mathcal L_\\infty\\).',
    ],
    tex: `${Linf}(${x1}) = \\tfrac12\\int_{${lmin}}^{${lmax}}\\E_{${x0}}\\big\\|${x0} - ${hx0}(\\x_{${lam}}, ${lam})\\big\\|^2\\,\\dd${lam} = \\int_{${lmin}}^{${lmax}} ${ell}(${lam})\\,\\dd${lam}`,
    deriv: {
      have: ['The denoising terms of step 2 on a grid \\(t_0 < \\dots < t_{N-1}\\): \\(\\tfrac12\\big(\\mathrm{SNR}(t_i) - \\mathrm{SNR}(t_{i-1})\\big)\\,\\E\\|\\x_1 - \\hx_1(\\x_{t_{i-1}}, t_{i-1})\\|^2\\).', '\\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) and \\(\\mathrm{SNR}(t) = e^{\\lambda_t}\\).'],
      goal: 'step',
      plan: 'Let the grid become fine so that the sum becomes an integral in t. Change the variable to λ, and turn the data-prediction error into a noise-prediction error.',
      title: 'From the sum to the integral', lines: [
      { tex: `\\sum_{i=1}^{${N}-1}\\tfrac12\\big(${snr}(t_i) - ${snr}(t_{i-1})\\big)\\,\\E\\|${x1} - ${hx1}\\|^2 \\;\\longrightarrow\\; ${CHG(`\\tfrac12\\int_0^1 \\frac{\\dd}{\\dd t}e^{\\lambda_t}\\;\\E\\|${x1} - ${hx1}(${xt}, t)\\|^2\\,\\dd t`)}`, why: 'Each term is an increment of \\(\\mathrm{SNR}\\) times the error at the left end of the step. As the largest step tends to zero, the sum tends to the integral of the error against \\(\\dd\\,\\mathrm{SNR}\\).', uses: ['elbo-step', 'riemann'] },
      { tex: `= \\tfrac12\\int_{${lmin}}^{${lmax}} ${CHG(`e^{${lam}}`)}\\;\\E\\|${x1} - ${hx1}(\\x_{${lam}}, ${lam})\\|^2\\,${CHG(`\\dd${lam}`)}`, why: 'Change variables from \\(t\\) to \\(\\lambda = \\lambda_t\\). Then \\(\\frac{\\dd}{\\dd t}e^{\\lambda_t}\\,\\dd t = e^{\\lambda}\\,\\dd\\lambda\\), and the limits become the log-SNR at the two ends.', uses: ['substitution', 'log-deriv'] },
      { tex: `${x0} - ${hx0} = \\frac{${xt} - \\alpha_t${x1}}{\\sigma_t} - \\frac{${xt} - \\alpha_t${hx1}}{\\sigma_t} = ${CHG(`-e^{${lam}/2}\\big(${x1} - ${hx1}\\big)`)}`, why: 'Solve the interpolation for \\(\\x_0\\) and take conditional expectations given \\(\\x_t\\); both are linear. The ratio \\(\\alpha_t/\\sigma_t\\) is \\(e^{\\lambda/2}\\).', uses: ['cond-exp'] },
      { tex: `${Linf} = \\tfrac12\\int_{${lmin}}^{${lmax}} \\E\\big\\|${x0} - ${hx0}(\\x_{${lam}}, ${lam})\\big\\|^2\\,\\dd${lam}`, why: 'So \\(e^{\\lambda}\\|\\x_1 - \\hx_1\\|^2 = \\|\\x_0 - \\hx_0\\|^2\\). The weight \\(e^\\lambda\\) of the data error is exactly the conversion to the noise error.', uses: [] },
    ] },
    after: ['For a smooth density \\(\\ell(\\lambda)\\) tends to \\(d/2\\) at high log-SNR: the noise becomes too small to be told apart from the data, so it cannot be predicted at all. The integral therefore grows by \\(\\tfrac12\\) per unit of \\(\\lambda_{\\max}\\), and the reconstruction term falls by the same amount. The full bound depends on \\(\\lambda_{\\max}\\) only through how well the last decoder fits.'],
    check: {
      q: 'Why do the bars of a coarse grid lie above the curve rather than below it?',
      opts: [['Each step uses the error at its noisier end, and the error in data coordinates is largest there', true], ['The bars include the prior term', false], ['Because the KL is always larger than the squared error', false]],
      fb: { true: 'Right. \\(\\E\\|\\x_1 - \\hx_1\\|^2\\) decreases with \\(\\lambda\\), and step 2 evaluates it at the start of each step. The bar therefore overestimates the integral over that step, and the overestimate vanishes as the steps shrink.', false: 'Step 2 evaluates the data error at the noisier end \\(\\x_s\\) of each step, and \\(\\E\\|\\x_1 - \\hx_1\\|^2\\) is a decreasing function of \\(\\lambda\\). A left Riemann sum of a decreasing function lies above the integral.' },
    },
  },
  {
    id: 'schedule', kicker: 'Invariance', title: 'The schedule changes only the variance', scene: 'schedule',
    prose: [
      'A training step does not compute the integral of step 3. It draws one time \\(t\\) uniformly, maps it to \\(\\lambda_t\\) through the noise schedule, and uses \\(\\dot\\lambda_t\\,\\ell(\\lambda_t)\\) as an unbiased estimate of \\(\\mathcal L_\\infty\\). Two schedules with the same ends define the same objective: the rescaled input \\(\\x_t/\\alpha_t = \\x_1 + e^{-\\lambda_t/2}\\x_0\\) depends on \\(t\\) only through \\(\\lambda_t\\), so the integrand is a fixed function of \\(\\lambda\\), and only the speed at which a schedule moves along it changes.',
      'The upper panel shows the integrand in time, \\(\\dot\\lambda_t\\,\\ell(\\lambda_t)\\), for four schedules with \\(\\lambda\\) from −9 to 9. The areas under the curves are equal. The lower panel shows the running mean of 2000 one-sample estimates per schedule, with the same draws of \\(t\\) for all of them. The data and noise are averaged exactly, so only the draw of \\(t\\) is random. Press "Play" or move the slider. All four converge to \\(\\mathcal L_\\infty\\), at different speeds. The last schedule moves through \\(\\lambda\\) at the rate \\(1/\\ell\\), which makes the integrand constant, and its estimate has no variance at all.',
    ],
    tex: `\\frac{${xt}}{\\alpha_t} = ${x1} + e^{-${S('sched')}/2}\\,${x0} \\quad\\Rightarrow\\quad \\int_0^1 ${S('lamdot')}\\,${ell}(${S('sched')})\\,\\dd t = \\int_{${lmin}}^{${lmax}} ${ell}(${lam})\\,\\dd${lam} = ${Linf}`,
    deriv: {
      have: ['Two increasing schedules \\(\\lambda_t\\) and \\(\\tilde\\lambda_t\\) with the same values \\(\\lambda_{\\min}\\) at \\(t = 0\\) and \\(\\lambda_{\\max}\\) at \\(t = 1\\).', 'A network that sees the rescaled input and the log-SNR: \\(\\hx_0(\\x, t) = f_\\theta(\\x/\\alpha_t, \\lambda_t)\\).'],
      goal: `\\int_0^1 ${S('lamdot')}\\,${ell}(${S('sched')})\\,\\dd t = ${Linf} \\text{ for every schedule}, \\qquad \\mathrm{Var}_t = \\int ${S('lamdot')}\\,${ell}(${lam})^2\\,\\dd${lam} - ${Linf}^2`,
      plan: 'Show that the noisy input, and so the error, depends on the time only through λ. Then change variables in the mean and the second moment of the one-sample estimate.',
      title: 'Why the schedule cancels', lines: [
      { tex: `\\frac{${xt}}{\\alpha_t} = ${x1} + \\frac{\\sigma_t}{\\alpha_t}\\,${x0} = ${x1} + ${CHG(`e^{-\\lambda_t/2}`)}\\,${x0}`, why: 'Divide the interpolation by \\(\\alpha_t\\) and use \\(\\lambda_t = \\log(\\alpha_t^2/\\sigma_t^2)\\).', uses: ['log-rules'] },
      { tex: `\\tfrac12\\E\\big\\|${x0} - f_\\theta(${xt}/\\alpha_t, \\lambda_t)\\big\\|^2 = ${CHG(`${ell}(\\lambda_t)`)}`, why: 'The law of \\((\\x_0, \\x_t/\\alpha_t)\\) depends on \\(t\\) only through \\(\\lambda_t\\), and so does the input of the network. The expected error is a fixed function of \\(\\lambda\\), the same for every schedule.', uses: ['expectation'] },
      { tex: `\\E_{t\\sim U[0,1]}\\big[\\dot\\lambda_t\\,${ell}(\\lambda_t)\\big] = \\int_0^1 \\dot\\lambda_t\\,${ell}(\\lambda_t)\\,\\dd t = ${CHG(`\\int_{${lmin}}^{${lmax}} ${ell}(${lam})\\,\\dd${lam}`)}`, why: 'Change variables from \\(t\\) to \\(\\lambda = \\lambda_t\\). Only the ends of the schedule remain.', uses: ['substitution'] },
      { tex: `\\E_{t}\\big[(\\dot\\lambda_t\\,${ell}(\\lambda_t))^2\\big] = \\int_0^1 \\dot\\lambda_t^2\\,${ell}(\\lambda_t)^2\\,\\dd t = ${CHG(`\\int_{${lmin}}^{${lmax}} \\dot\\lambda\\;${ell}(${lam})^2\\,\\dd${lam}`)}`, why: 'The same substitution in the second moment leaves one factor \\(\\dot\\lambda\\), now read as a function of \\(\\lambda\\). The variance depends on the schedule through it, and it is zero when \\(\\dot\\lambda\\,\\ell\\) is constant, that is when \\(\\dot\\lambda \\propto 1/\\ell(\\lambda)\\).', uses: ['substitution'] },
    ] },
    after: ['VDM learns the schedule to minimize this variance. With a real network \\(\\ell\\) is unknown, so it is estimated during training, but the target is the same flat integrand as in the last schedule here.'],
    check: {
      q: 'The linear path \\(\\alpha_t = t\\), \\(\\sigma_t = 1-t\\) is used with \\(\\lambda\\) between the same ends. Compared with the log-SNR linear schedule, what does it change?',
      opts: [['Only the variance of the estimate; the objective is the same', true], ['It gives a different objective with a different minimizer', false], ['Nothing at all, not even the variance', false]],
      fb: { true: 'Right. The two curves enclose the same area, and their running means reach the same value. The linear path spends little time at the extremes of \\(\\lambda\\), where its \\(\\dot\\lambda_t\\) is large, which gives occasional very large estimates and a larger standard deviation.', false: 'Compare the readout: the means agree to the fourth digit, but the standard deviations of the one-sample estimates differ. The schedule enters only through \\(\\dot\\lambda\\) in the second moment.' },
    },
  },
  {
    id: 'weights', kicker: 'Weighting', title: 'Every loss is a weighting of one integrand', scene: 'weights',
    prose: [
      'The losses used in practice are not the ELBO, but each of them is the integrand \\(\\ell(\\lambda)\\) with a different weight \\(w(\\lambda)\\), the formula below. The weight collects two choices: the density \\(\\pi(t)\\) from which training times are drawn, and the prediction target, which multiplies the error by a factor that depends on the level. For flow matching on the linear path both are visible in the second formula; the derivation computes it.',
      'Select a weighting. The upper panel shows \\(w\\) on a log scale, and the lower panel shows the weighted integrand \\(w\\,\\ell\\), whose area is the loss. Each weighting is scaled so that this area equals \\(\\mathcal L_\\infty\\) for the exact denoiser, which makes the shapes comparable. The ELBO weights every level equally. Uniform flow matching weights the noisy end more by a factor of up to \\(e^{9}\\). The logit-normal time sampling of Stable Diffusion 3 and the log-normal noise levels of EDM concentrate the weight on intermediate levels.',
    ],
    tex: `\\mathcal L_{${w}} = \\int ${w}(${lam})\\,${ell}(${lam})\\,\\dd${lam}, \\qquad ${w}(${lam}) = ${pi}(t)\\,\\frac{1-t}{t} = ${pi}(t)\\,e^{-${lam}/2} \\quad\\text{for flow matching on the linear path}`,
    deriv: {
      have: ['The flow-matching loss \\(\\E_{t\\sim\\pi}\\,\\E\\|\\hat\\vel - \\vel\\|^2\\) with the linear path \\(\\x_t = t\\,\\x_1 + (1-t)\\,\\x_0\\) and \\(\\vel = \\x_1 - \\x_0\\).', 'The model velocity written through a noise prediction, \\(\\hat\\vel = \\hx_1 - \\hx_0\\).'],
      goal: `\\E_{t\\sim${pi}}\\,\\E\\big\\|${hu} - ${u}\\big\\|^2 = \\int ${pi}(t)\\,e^{-${lam}/2}\\;${ell}(${lam})\\,\\dd${lam}`,
      plan: 'Express the velocity error as a noise error, then change the variable from t to λ.',
      title: 'The weight of flow matching', lines: [
      { tex: `${u} = ${x1} - ${x0} = ${CHG(`\\frac{${xt} - ${x0}}{t}`)}, \\qquad ${hu} = \\frac{${xt} - ${hx0}}{t}`, why: 'Solve the interpolation for \\(\\x_1 = (\\x_t - (1-t)\\x_0)/t\\) and subtract \\(\\x_0\\). The model velocity is the same expression with the predicted noise.', uses: ['cond-exp'] },
      { tex: `\\big\\|${hu} - ${u}\\big\\|^2 = ${CHG(`\\frac{1}{t^2}`)}\\,\\big\\|${x0} - ${hx0}\\big\\|^2`, why: 'The noisy point cancels in the difference.', uses: [] },
      { tex: `${lam} = 2\\log\\frac{t}{1-t}, \\qquad ${CHG(`\\frac{\\dd${lam}}{\\dd t} = \\frac{2}{t(1-t)}`)}`, why: 'The log-SNR of the linear path is \\(\\log(t^2/(1-t)^2)\\). Differentiate the log of the ratio.', uses: ['log-rules', 'log-deriv'] },
      { tex: `\\int_0^1 ${pi}(t)\\,\\frac{1}{t^2}\\,2${ell}(${lam})\\,\\dd t = \\int ${pi}(t)\\,\\frac{1}{t^2}\\,2${ell}(${lam})\\;${CHG('\\frac{t(1-t)}{2}')}\\,\\dd${lam} = \\int ${pi}(t)\\,${CHG('\\frac{1-t}{t}')}\\,${ell}(${lam})\\,\\dd${lam}`, why: 'Write the expectation over \\(t\\) as an integral and change variables to \\(\\lambda\\), with \\(\\dd t = \\tfrac12 t(1-t)\\,\\dd\\lambda\\).', uses: ['substitution'] },
      { tex: `${w}(${lam}) = ${pi}(t)\\,\\frac{1-t}{t} = ${pi}(t)\\,${CHG(`e^{-${lam}/2}`)}`, why: 'From the definition of \\(\\lambda\\), \\(t/(1-t) = e^{\\lambda/2}\\).', uses: ['log-rules'] },
    ] },
    after: ['Measured against the ELBO, all of these weightings discount the least noisy levels, where the error concerns details that are hard to see, and spend capacity on the levels that decide global structure. With the exact denoiser the weighting does not matter, because every term is at its minimum. With a network of limited size it decides which errors are fixed first, which is why the choice affects sample quality.'],
    check: {
      q: 'Relative to the ELBO, how much more does uniform flow matching weight the noisiest level here than the cleanest?',
      opts: [['About \\(e^{9}\\), roughly 8000 times', true], ['About 2 times', false], ['The same: uniform t means uniform weight', false]],
      fb: { true: 'Right. \\(w = e^{-\\lambda/2}\\) for uniform \\(\\pi\\), so the ratio between \\(\\lambda = -9\\) and \\(\\lambda = 9\\) is \\(e^{9} \\approx 8100\\). Uniform in t is far from uniform in log-SNR.', false: 'With uniform \\(\\pi\\), \\(w(\\lambda) = e^{-\\lambda/2}\\). Between \\(\\lambda = -9\\) and \\(\\lambda = 9\\) it falls by \\(e^{9} \\approx 8100\\). The upper panel uses a log scale for this reason.' },
    },
  },
  {
    id: 'monotone', kicker: 'A likelihood reading of weights', title: 'Decreasing weights are ELBOs of noisy data', scene: 'monotone',
    prose: [
      'Kingma and Gao showed when a weighted loss still has a likelihood meaning. Perturb the data with Gaussian noise at level \\(\\lambda\'\\) and treat the noisy point as the data. Its ELBO involves only the noisier levels, so its loss is the part of the integral below \\(\\lambda\'\\). Averaging over \\(\\lambda\'\\) with a density \\(p\\) and exchanging the integrals gives a weighted loss with \\(w(\\lambda) = \\int_\\lambda^{\\lambda_{\\max}} p\\). Read backwards: a weight that decreases in \\(\\lambda\\) is maximum likelihood with Gaussian noise augmentation, with \\(p = -w\'\\).',
      'Select a weighting. The upper panel shows \\(w\\), and the lower panel \\(-w\'\\). Where \\(-w\'\\) is negative the weight increases, \\(-w\'\\) is not a density, and the region is shaded. The readout checks the identity of the formula numerically: it holds for every weight, but only a weight without shaded regions is an average of ELBOs.',
    ],
    tex: `\\int ${w}(${lam})\\,${ell}(${lam})\\,\\dd${lam} = \\int \\big(-${w}'(${lp})\\big)\\underbrace{\\int_{${lmin}}^{${lp}} ${ell}(${lam})\\,\\dd${lam}}_{\\text{ELBO of data noised to } ${lp}}\\,\\dd${lp} \\;+\\; ${w}(${lmax})\\,${Linf}`,
    deriv: {
      have: ['A weight \\(w\\) on \\([\\lambda_{\\min}, \\lambda_{\\max}]\\) with derivative \\(w\'\\).', 'The loss of the ELBO for data noised to level \\(\\lambda\'\\): \\(\\int_{\\lambda_{\\min}}^{\\lambda\'}\\ell(\\lambda)\\,\\dd\\lambda\\), up to terms without θ.'],
      goal: 'step',
      plan: 'Write w as the integral of its derivative, insert it in the weighted loss, and exchange the order of the two integrals.',
      title: 'Why decreasing weights are averages of ELBOs', lines: [
      { tex: `${w}(${lam}) = \\int_{${lam}}^{${lmax}} \\big(-${w}'(${lp})\\big)\\,\\dd${lp} + ${w}(${lmax})`, why: 'The weight at \\(\\lambda\\) is its value at the top end plus everything it loses on the way up.', uses: ['ftc'] },
      { tex: `\\int ${w}\\,${ell}\\,\\dd${lam} = \\int ${ell}(${lam})${CHG(`\\int_{${lam}}^{${lmax}} \\big(-${w}'(${lp})\\big)\\,\\dd${lp}`)}\\,\\dd${lam} + ${w}(${lmax})\\,${Linf}`, why: 'Insert into the weighted loss. The constant part multiplies the whole integral of \\(\\ell\\).', uses: [] },
      { tex: `= \\int \\big(-${w}'(${lp})\\big)${CHG(`\\int_{${lmin}}^{${lp}} ${ell}(${lam})\\,\\dd${lam}`)}\\,\\dd${lp} + ${w}(${lmax})\\,${Linf}`, why: 'Both double integrals run over the pairs with \\(\\lambda \\le \\lambda\'\\). Exchange the order: for fixed \\(\\lambda\'\\), \\(\\lambda\\) runs from \\(\\lambda_{\\min}\\) up to \\(\\lambda\'\\).', uses: ['fubini'] },
      { tex: `-${w}' \\ge 0 \\;\\Rightarrow\\; \\mathcal L_{${w}} = \\E_{${lp}\\sim -${w}'}\\big[\\text{ELBO loss of data noised to } ${lp}\\big] + ${w}(${lmax})\\,${Linf}`, why: 'If the weight never increases, \\(-w\'\\) is a non-negative density (up to its total mass), and the loss is an average of ELBOs plus a multiple of the ELBO of the clean data.', uses: [] },
    ] },
    after: ['Uniform flow matching passes this test, and it is maximum likelihood on noise-augmented data. Bump-shaped weights such as those of SD3 and EDM fall outside it: they increase over part of the range, so the losses are not averages of likelihoods, although they often give better samples.'],
    check: {
      q: 'Select the EDM weighting. Why is it not an average of ELBOs?',
      opts: [['It increases at the noisy end, so \\(-w\'\\) is negative there', true], ['It is not normalized', false], ['Its value at \\(\\lambda_{\\max}\\) is positive', false]],
      fb: { true: 'Right. The log-normal density of EDM noise levels falls to zero at very high noise, so \\(w\\) rises before it falls. A rising weight needs a negative density of noise levels, which does not exist.', false: 'Scaling does not matter, and a positive value at \\(\\lambda_{\\max}\\) only adds the ELBO of the clean data. The shaded region of the lower panel is the problem: where \\(w\\) increases, \\(-w\'\\) is negative and cannot be a density.' },
    },
  },
  {
    id: 'exact', kicker: 'Beyond the bound', title: 'Exact likelihood through the probability-flow ODE', scene: 'exact',
    prose: [
      'The ELBO is a bound for the stochastic sampler. The deterministic sampler has an exact likelihood. Along a path of the ODE \\(\\dot\\x_t = \\bm u_t(\\x_t)\\), the continuity equation of §2.4 implies that the log-density changes at the rate minus the divergence of the velocity, the formula below. Integrating from noise to data gives \\(\\log p_1(\\x_1)\\) from the noise end of the path, which is found by running the ODE backwards from \\(\\x_1\\).',
      'The plot does this for the 1D mixture with the exact marginal velocity of the linear path. The lower panel shows the backward paths of a few data points; move the cursor to choose the probe \\(\\x_1\\), and click to pin it. The upper panel shows the exact \\(\\log\\pdata\\) and the ODE estimates at the same points. Move \\(K\\), the number of RK4 steps. With two steps the estimates are visibly off; from about eight they agree with the exact curve to three digits. In \\(d\\) dimensions the divergence is a trace that costs \\(d\\) backward passes, so it is replaced by Hutchinson’s estimator \\(\\diver\\vel = \\E_{\\bm\\epsilon}\\big[\\bm\\epsilon^\\top(\\partial\\vel/\\partial\\x)\\bm\\epsilon\\big]\\), which needs one vector–Jacobian product per sample.',
    ],
    tex: `\\frac{\\dd}{\\dd t}\\log ${pt}(${xt}) = -${div}${u}_t(${xt}), \\qquad \\log p_1(${x1}) = \\log p_0(${x0}) - \\int_0^1 ${div}${u}_t(${xt})\\,\\dd t`,
    deriv: {
      have: ['A path \\(\\x_t\\) of the ODE \\(\\dot\\x_t = \\bm u_t(\\x_t)\\), where \\(\\bm u_t\\) generates the marginals \\(p_t\\).', 'The continuity equation \\(\\partial_t p_t + \\diver(p_t\\bm u_t) = 0\\) of §2.4.'],
      goal: 'step',
      plan: 'Differentiate the log-density along the path, use the continuity equation for its time derivative, and integrate over time.',
      title: 'The instantaneous change of variables', lines: [
      { tex: `\\frac{\\dd}{\\dd t}\\log ${pt}(${xt}) = \\partial_t\\log ${pt} + ${u}_t\\cdot\\nabla\\log ${pt}`, why: 'The chain rule: the density changes at a fixed point, and the point moves with velocity \\(\\bm u_t\\).', uses: ['chain-rule'] },
      { tex: `\\partial_t ${pt} = -${div}(${pt}${u}_t) = ${CHG(`-${pt}\\,${div}${u}_t - ${u}_t\\cdot\\nabla ${pt}`)}`, why: 'The continuity equation, with the divergence of a product expanded by the product rule.', uses: ['continuity', 'div-grad-lap'] },
      { tex: `\\partial_t\\log ${pt} = \\frac{\\partial_t ${pt}}{${pt}} = ${CHG(`-${div}${u}_t - ${u}_t\\cdot\\nabla\\log ${pt}`)}`, why: 'Divide by \\(p_t\\), using \\(\\nabla p_t / p_t = \\nabla\\log p_t\\).', uses: ['log-deriv'] },
      { tex: `\\frac{\\dd}{\\dd t}\\log ${pt}(${xt}) = ${CHG(`-${div}${u}_t(${xt})`)}`, why: 'Insert into the first line. The transport terms cancel, and only the divergence remains.', uses: [] },
      { tex: `\\log p_1(${x1}) = \\log p_0(${x0}) - \\int_0^1 ${div}${u}_t(${xt})\\,\\dd t`, why: 'Integrate from \\(t = 0\\) to \\(t = 1\\). The noise end \\(\\x_0\\) is found by solving the ODE backwards from \\(\\x_1\\), and \\(p_0 = \\N(0, I)\\).', uses: ['ftc'] },
    ] },
    after: ['This is the change of variables of continuous normalizing flows, and it is how score-based models report bits per dimension. The value belongs to the deterministic sampler. It is in general not the likelihood of the reverse SDE, and with a learned field it is the exact likelihood of the model, not of the data.'],
    check: {
      q: 'Hutchinson’s estimator uses random vectors \\(\\bm\\epsilon\\) with \\(\\E[\\bm\\epsilon\\bm\\epsilon^\\top] = I\\). Why is \\(\\E[\\bm\\epsilon^\\top A\\bm\\epsilon]\\) the trace of \\(A\\)?',
      opts: [['\\(\\E[\\bm\\epsilon^\\top A\\bm\\epsilon] = \\sum_{ij}A_{ij}\\E[\\epsilon_i\\epsilon_j] = \\sum_i A_{ii}\\)', true], ['Because \\(A\\) is symmetric', false], ['Only if \\(\\bm\\epsilon\\) is Gaussian', false]],
      fb: { true: 'Right. Only the diagonal survives the expectation, and it works for any distribution with identity covariance, for example random signs. One sample costs one vector–Jacobian product instead of d.', false: 'Expand the quadratic form: \\(\\bm\\epsilon^\\top A\\bm\\epsilon = \\sum_{ij}A_{ij}\\epsilon_i\\epsilon_j\\). Its expectation keeps only \\(i = j\\), because \\(\\E[\\epsilon_i\\epsilon_j] = \\delta_{ij}\\). Neither symmetry nor Gaussianity is needed.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>§2.5 of the survey continues with objectives that change the target instead of the weight: flow maps, consistency models and MeanFlow, which learn the result of following the field over an interval. They are the subject of §6. The weighting of this tutorial returns in §3 and §4, where fine-tuning and reward methods choose which noise levels to train on.</p>
      <p>The earlier tutorials, <a href="../score-matching/">Learning the score</a>, <a href="../sde/">Diffusion as an SDE</a> and <a href="../flow-matching/">Flows and flow matching</a>, derive the regression losses, the probability-flow ODE and the continuity equation used here.</p>`,
  },
];
