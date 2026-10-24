// Content of the §4 tutorial: one entry per step. Formulas follow paper/sections/04_posttraining.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), pd = S('pd'), r = S('r'), beta = S('beta'), ps = S('ps'), Z = S('Z'), rh = S('rh'), u = S('u'), sc = S('s');
const eps = S('eps'), lam = S('lam'), theta = S('theta'), pth = S('pth'), pi = S('pi'), A = S('A'), K = S('K'), xw = S('xw'), xl = S('xl'), h = S('h');

export const INTRO = `
  <div class="kicker">Survey §4 · Interactive tutorial</div>
  <h1>Reward fine-tuning and preference alignment</h1>
  <p>A pretrained diffusion or flow model samples its training distribution. Post-training changes it so that samples score higher under a reward, without drifting far from what the model already does well. This tutorial starts from the density that such training should produce, the base model tilted by the exponentiated reward, and then follows the main ways of getting there: policy gradients on the chain of denoising steps, gradients through the sampler, preference pairs without an explicit reward, and stochastic optimal control. Along the way it shows why a reward model fitted to few labels is dangerous, why reinforcement learning needs a stochastic sampler, and why the noise of that sampler has to be chosen with care.</p>
  <p>The base model is the exact three-mode mixture of §2.1 on the real line, so it has no error of its own. The reward is \\(r(\\x) = -(\\x - a)^2/2\\), which keeps every tilted target a Gaussian mixture with closed-form parameters. The model that is trained is a mixture of three Gaussians with nine free parameters, sampled with the exact fields of its own noised mixture, so its density and every KL on the page are exact at each iteration. Training runs in the page from fixed seeds. The convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\).</p>
  <ol class="toc">
    <li><a href="#tilt">What reward fine-tuning should produce</a></li>
    <li><a href="#proxy">Proxy rewards and over-optimization</a></li>
    <li><a href="#sde">A sampler with a likelihood</a></li>
    <li><a href="#pg">Policy gradient on the denoising chain</a></li>
    <li><a href="#pathwise">Backpropagating through the sampler</a></li>
    <li><a href="#dpo">Preferences without a reward</a></li>
    <li><a href="#control">Why the noise must be memoryless</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'tilt', kicker: 'The target', title: 'What reward fine-tuning should produce', scene: 'tilt',
    prose: [
      'Maximizing the reward alone has a useless answer: a point mass on the best sample. Every practical method instead trades the reward against the distance to the base model, measured by the KL divergence with a weight β. Over all densities this objective has a closed-form maximizer, the base model multiplied by \\(e^{r/\\beta}\\) and normalized. It keeps the support of the base model and moves mass towards high reward, more strongly the smaller β is.',
      'For the quadratic reward the tilt multiplies each Gaussian component by another Gaussian, so the target is again a mixture of three Gaussians. Each component narrows and moves towards a, and its weight changes with how close it already was. Move β and a and watch the reward gained and the KL paid on the right.',
    ],
    tex: `\\max_p\\ \\E_{p}[${r}(${x})] - ${beta}\\,\\KL(p\\,\\|\\,${pd}) \\quad\\text{is solved by}\\quad ${ps}(${x}) = \\frac{${pd}(${x})\\,e^{${r}(${x})/${beta}}}{${Z}}`,
    deriv: {
      have: ['Any density p with \\(\\KL(p\\,\\|\\,\\pdata) < \\infty\\).', 'The normalizer \\(Z = \\int \\pdata(\\x)\\,e^{r(\\x)/\\beta}\\,\\dd\\x\\), assumed finite.'],
      goal: '\\E_p[r] - \\beta\\,\\KL(p\\,\\|\\,\\pdata) = -\\beta\\,\\KL(p\\,\\|\\,p^*) + \\beta\\log Z',
      plan: 'Write the KL as an expectation, pull the reward inside the logarithm, and recognize the KL to the tilted density.',
      title: 'The Gibbs variational principle', lines: [
        { tex: `\\E_p[r] - \\beta\\,\\KL(p\\,\\|\\,\\pdata) = ${CHG('-\\beta\\,\\E_p\\Big[\\log\\frac{p(\\x)}{\\pdata(\\x)} - \\frac{r(\\x)}{\\beta}\\Big]')}`, why: 'The KL is the expectation under p of the log ratio; both terms are expectations under p.', uses: ['kl', 'expectation'] },
        { tex: `= -\\beta\\,\\E_p\\Big[\\log\\frac{p(\\x)}{${CHG('\\pdata(\\x)\\,e^{r(\\x)/\\beta}')}}\\Big]`, why: 'r/β is the log of \\(e^{r/\\beta}\\), and a difference of logs is the log of a ratio.', uses: ['log-rules'] },
        { tex: `= -\\beta\\,\\E_p\\Big[\\log\\frac{p(\\x)}{${CHG('p^*(\\x)')}}\\Big] ${CHG('+ \\beta\\log Z')}`, why: 'Divide and multiply by Z inside the logarithm; log Z is a constant.', uses: ['log-rules', 'expectation'] },
        { tex: `= ${CHG('-\\beta\\,\\KL(p\\,\\|\\,p^*)')} + \\beta\\log Z`, why: 'The definition of the KL. It is zero exactly at p = p*, and positive otherwise.', uses: ['kl'] },
      ],
    },
    after: ['At a = 1.2 and β = 0.5 the target puts 0.49 and 0.51 of its mass on the two right modes, which it pulls to 0.48 and 1.67, and almost nothing on the left one. Its expected reward is −0.259 against −2.042 for the base model, at a KL of 0.724. At β = 10 it gains little (−1.671) and pays almost nothing (KL 0.018); at β = 0.1 it reaches −0.073 at a KL of 1.580. Every method on this page aims at this density, and the remaining steps measure how close each one gets.'],
    check: {
      q: 'What does \\(\\KL(p^*\\,\\|\\,\\pdata)\\) tend to as \\(\\beta \\to \\infty\\)?',
      answer: 0, tol: 0.01,
      fb: { true: 'Right. As β grows, \\(e^{r/\\beta}\\) tends to 1 everywhere, so the target is the base model and the KL vanishes. As β → 0 the target collapses onto the points of the support with the largest reward.', false: 'Look at the factor \\(e^{r(\\x)/\\beta}\\) for very large β. What density is left after normalizing?' },
    },
  },
  {
    id: 'proxy', kicker: 'Reward models', title: 'Proxy rewards and over-optimization', scene: 'proxy',
    prose: [
      'In practice the reward is itself a model, fitted to a limited set of human or automatic labels. It is accurate where the labels were collected and arbitrary elsewhere. Here the gold reward is a bump around 1.1, and the proxy is the straight line fitted by least squares to 20 noisy labels at draws from the base model. On the data the line follows the rising flank of the bump, but beyond it the line keeps rising while the gold reward falls.',
      'Tilting by the proxy with a small β moves mass to wherever the proxy is largest, which is the far right, outside the labelled region. The right plot follows the tilted target as β falls, and plots both rewards against the square root of the KL, the axis on which reward-model over-optimization is usually reported. Press New labels to fit the proxy to another seeded set of 20 labels.',
    ],
    tex: `${ps}_{\\beta} \\propto ${pd}\\,e^{${rh}/${beta}}, \\qquad \\text{reported: } \\E_{p^*_\\beta}[${r}] \\text{ and } \\E_{p^*_\\beta}[${rh}] \\text{ against } \\sqrt{\\KL(p^*_\\beta\\,\\|\\,${pd})}`,
    after: ['With the first set of labels the gold reward of the tilted target rises from 0.224 for the base model to 0.319 at β ≈ 0.25, a square-root KL of 0.63, and then falls to 0.018 at β = 0.032. Over the same range the proxy reward rises from 0.241 to 0.751 without a dip. The proxy cannot see the failure, so the stopping point must come from somewhere else: a held-out gold metric, a larger β, or a reward model that is retrained on samples of the current policy.'],
    check: {
      q: 'Why does the gold reward rise and then fall while the proxy reward keeps rising?',
      opts: [
        ['At large β the tilt stays where the proxy was fitted and agrees with the gold reward; at small β it moves mass beyond the labelled region, where the proxy keeps growing and the gold reward does not.', true],
        ['The tilted target has a larger KL at small β, and a larger KL lowers the gold reward by definition.', false],
        ['The proxy is fitted with label noise, and the noise grows as β falls.', false],
      ],
      fb: { true: 'Right. The KL term is the only thing that keeps the target where the proxy was fitted. A larger KL is not harmful in itself: with the gold reward as the tilt, the reward would keep rising.', false: 'Compare where the tilted target puts its mass at β = 1 and at β = 0.05 with where the 20 labels are.' },
    },
  },
  {
    id: 'sde', kicker: 'Stochastic samplers', title: 'A sampler with a likelihood', scene: 'sde',
    prose: [
      'The methods of the next steps treat sampling as a sequence of random decisions and need the probability of each decision. The probability-flow ODE of the base model makes none: given the start, the path is fixed. A whole family of stochastic samplers has the same marginals \\(p_t\\) as the ODE. Each adds noise of level \\(\\varepsilon_t\\) and a score term that undoes the spread the noise causes on average, so all members end at the data distribution.',
      'With the Euler–Maruyama discretization each step is a Gaussian with a known mean and variance, and the log-density of a path is the sum of the step log-densities. The tutorial scales the noise as a multiple λ of the memoryless level \\(\\varepsilon_t^2 = 2(1-t)/t\\), which step 7 explains. Move λ: the paths from one start spread out, the end histogram stays on the data density, and the per-step log-probabilities appear once λ > 0.',
    ],
    tex: `\\dd${x}_t = \\Big[${u}(${x}_t) + \\frac{${eps}^2}{2}\\,${sc}(${x}_t)\\Big]\\dd t + ${eps}\\,\\dd W_t, \\qquad ${eps}^2 = ${lam}^2\\,\\frac{2(1 - t)}{t}`,
    deriv: {
      have: ['The continuity equation of the ODE, \\(\\partial_t p_t = -\\partial_x(u_t\\,p_t)\\).', 'The Fokker–Planck equation of an SDE with drift b and noise level ε: \\(\\partial_t p_t = -\\partial_x(b\\,p_t) + \\tfrac{\\varepsilon^2}{2}\\,\\partial_x^2 p_t\\).'],
      goal: '\\partial_t p_t = -\\partial_x(u_t\\,p_t) \\text{ for every } \\varepsilon_t',
      plan: 'Insert the drift of the family into the Fokker–Planck equation and use \\(p\\,\\partial_x\\log p = \\partial_x p\\).',
      title: 'The added drift cancels the added noise', lines: [
        { tex: `\\partial_t p_t = -\\partial_x\\Big(${CHG('\\big[u_t + \\tfrac{\\varepsilon^2}{2}\\,\\partial_x\\log p_t\\big]')}\\,p_t\\Big) + \\frac{\\varepsilon^2}{2}\\,\\partial_x^2 p_t`, why: 'The Fokker–Planck equation with the drift of the family.', uses: ['fokker-planck'] },
        { tex: `= -\\partial_x(u_t\\,p_t) - \\frac{\\varepsilon^2}{2}\\,\\partial_x(${CHG('\\partial_x p_t')}) + \\frac{\\varepsilon^2}{2}\\,\\partial_x^2 p_t`, why: 'The derivative of the log times the density is the derivative of the density.', uses: ['log-deriv'] },
        { tex: `= ${CHG('-\\partial_x(u_t\\,p_t)')}`, why: 'The two diffusion terms cancel, and what is left is the continuity equation of the ODE.', uses: ['continuity'] },
      ],
    },
    after: ['The end points of 2000 paths are at a Wasserstein distance of 0.031 from the data for the ODE, and 0.035, 0.026, 0.030 and 0.043 for λ = 0.25, 0.5, 1 and 1.5. Two thousand exact draws give 0.038 on average, so every sampler in the family ends at the data up to sampling noise. The paths differ: the summed step log-density of the orange path is 54.9 at λ = 0.25 and −16.7 at λ = 1.5, because small steps of small noise have large densities. Only the differences of these numbers between two models matter, and they are what the next step differentiates.'],
    check: {
      q: 'Why does the deterministic sampler give policy-gradient methods nothing to differentiate?',
      opts: [
        ['Each step is a point mass at the ODE update, so a path has no density, and the log-probability of a step is not defined.', true],
        ['The ODE has a likelihood, but it equals the likelihood of the data, which does not depend on the parameters.', false],
        ['The ODE ends at a different distribution than the SDE, so its rewards are biased.', false],
      ],
      fb: { true: 'Right. The end point of the ODE does have a density, by the change-of-variables formula, but it needs the divergence of the velocity along the whole path. The stochastic sampler replaces it with a sum of Gaussian log-densities.', false: 'Look at the right panel at λ = 0. What is the probability of the next point given the current one?' },
    },
  },
  {
    id: 'pg', kicker: 'Reinforcement learning', title: 'Policy gradient on the denoising chain', scene: 'pg',
    prose: [
      'With a stochastic sampler, generating a sample is a sequence of decisions, each a Gaussian whose mean depends on the parameters. The gradient of the expected reward is then the expected reward times the gradient of the log-density of the path, which is a sum over the steps. DDPO uses this estimator with a running baseline. Flow-GRPO and DanceGRPO draw a group of G rollouts from a shared start, normalize the rewards within the group, and reuse each batch for a few clipped steps, as PPO does.',
      'The return here is the reward minus β times the log-ratio of the path densities of the trained model and the base model, so the expected return is the objective of step 1 on paths. Press Play: the colours on the left show which rollouts of the first group push the model and which pull it, the middle panel shows the model approaching the target, and the curves on the right show reward and KL settling at the values of p*, marked by dotted lines.',
    ],
    tex: `\\nabla_{${theta}} J = \\E\\Big[${A}\\sum_{k} \\nabla_{${theta}}\\log ${pi}(\\x_{k+1}\\mid\\x_k)\\Big], \\qquad ${A} = \\frac{R - \\mathrm{mean}_G(R)}{\\mathrm{std}_G(R)}, \\quad R = ${r}(\\x_1) - ${beta}\\log\\frac{${pth}(\\text{path})}{p_{\\mathrm{ref}}(\\text{path})}`,
    deriv: {
      have: ['A path \\(\\tau = (\\x_0, \\dots, \\x_K)\\) with density \\(p_\\theta(\\tau) = p(\\x_0)\\prod_k \\pi_\\theta(\\x_{k+1}\\mid\\x_k)\\).', 'A return R(τ) that does not depend on θ.'],
      goal: '\\nabla_\\theta\\,\\E_{p_\\theta}[R] = \\E_{p_\\theta}\\Big[(R - b)\\sum_k \\nabla_\\theta\\log\\pi_\\theta(\\x_{k+1}\\mid\\x_k)\\Big] \\text{ for any constant } b',
      plan: 'Differentiate under the integral, write the gradient of the density through its logarithm, split the log of the product, and show that a constant baseline adds nothing.',
      title: 'The score-function identity on paths', lines: [
        { tex: `\\nabla_\\theta\\,\\E_{p_\\theta}[R] = ${CHG('\\int R(\\tau)\\,\\nabla_\\theta p_\\theta(\\tau)\\,\\dd\\tau')}`, why: 'Write the expectation as an integral and move the gradient inside.', uses: ['diff-under-int'] },
        { tex: `= \\int R(\\tau)\\,${CHG('p_\\theta(\\tau)\\,\\nabla_\\theta\\log p_\\theta(\\tau)')}\\,\\dd\\tau = \\E_{p_\\theta}\\big[R\\,\\nabla_\\theta\\log p_\\theta(\\tau)\\big]`, why: 'The gradient of a density is the density times the gradient of its log.', uses: ['log-deriv', 'expectation'] },
        { tex: `\\nabla_\\theta\\log p_\\theta(\\tau) = ${CHG('\\sum_k \\nabla_\\theta\\log\\pi_\\theta(\\x_{k+1}\\mid\\x_k)')}`, why: 'The log of a product is a sum, and the start density does not depend on θ.', uses: ['log-rules'] },
        { tex: `\\E_{p_\\theta}\\big[b\\,\\nabla_\\theta\\log p_\\theta(\\tau)\\big] = b\\,\\nabla_\\theta\\!\\int p_\\theta(\\tau)\\,\\dd\\tau = ${CHG('b\\,\\nabla_\\theta 1 = 0')}`, why: 'The same two steps backwards; a density integrates to one for every θ.', uses: ['diff-under-int', 'log-deriv'] },
      ],
    },
    after: ['With G = 16 and β = 0.5, GRPO brings the KL to the target from 2.84 down to 1.015 after 10 iterations, 0.109 after 30 and 0.003 after 100. DDPO with the same batch is slower, at 1.710, 0.616 and 0.013 at the same iterations, because it takes one step per batch where GRPO takes two clipped steps. Both end at an expected reward of −0.265 and −0.267, against −0.259 for p*. The path KL of the last batches is about 0.75, a little above the end-point KL of 0.715, as it must be: the KL between two path distributions is never smaller than the KL between their end points. In the full derivation R also depends on θ through the log-ratio, but the extra term is the expected score, which the last line shows to be zero.'],
    check: {
      q: 'Every rollout of a group gets the same return. What is the GRPO update from that group?',
      answer: 0, tol: 0.001,
      fb: { true: 'Right. All advantages are zero, so the group contributes nothing. With a sharp reward and a small group this happens often, which is why groups are made large enough to contain both good and bad rollouts.', false: 'Compute the advantage \\((R - \\mathrm{mean})/\\mathrm{std}\\) when all R are equal. The implementation sets it to zero when the standard deviation is zero.' },
    },
  },
  {
    id: 'pathwise', kicker: 'Differentiable rewards', title: 'Backpropagating through the sampler', scene: 'pathwise',
    prose: [
      'When the reward is differentiable, as an aesthetic or preference model on images is, the gradient can go through the sample instead of through its probability. DRaFT, AlignProp and ReFL compute how the end point moves when the parameters move, by the chain rule through every sampler step, and multiply by the gradient of the reward. The sensitivity is carried forward with one recursion per step. Storing every step for backpropagation is expensive for a large network, so in practice the gradient is often cut after the last K steps.',
      'The left plot compares the relative error of three estimators of the same gradient at the base model as the batch grows: the score-function estimator of step 4, the full pathwise estimator, and the pathwise estimator truncated to the last K steps. The right plot shows the nine components of the full gradient and of the truncated one. Move K.',
    ],
    tex: `\\nabla_{${theta}} \\E[${r}(\\x_1)] = \\E\\Big[\\nabla ${r}(\\x_1)\\,\\frac{\\partial\\x_1}{\\partial${theta}}\\Big], \\qquad \\frac{\\partial\\x_{k+1}}{\\partial${theta}} = \\big(1 + h\\,\\partial_x b_{${theta}}\\big)\\frac{\\partial\\x_k}{\\partial${theta}} + h\\,\\partial_{${theta}} b_{${theta}}`,
    deriv: {
      have: ['The Euler–Maruyama step \\(\\x_{k+1} = \\x_k + h\\,b_\\theta(\\x_k, t_k) + \\sqrt{h}\\,\\varepsilon_k z_k\\) with noise \\(z_k\\) that does not depend on θ.', 'A start \\(\\x_0\\) that does not depend on θ.'],
      goal: '\\nabla_\\theta\\,\\E[r(\\x_1)] = \\E\\big[r\'(\\x_1)\\,J_K\\big], \\quad J_k = \\partial\\x_k/\\partial\\theta',
      plan: 'Fix the noise, so the end point is a differentiable function of θ, and apply the chain rule step by step.',
      title: 'The pathwise gradient', lines: [
        { tex: `\\nabla_\\theta\\,\\E[r(\\x_1)] = ${CHG('\\E\\big[\\nabla_\\theta\\, r(\\x_1(\\theta; \\x_0, z))\\big]')}`, why: 'With the noise drawn first, the expectation is over a distribution that does not depend on θ, and the gradient moves inside.', uses: ['diff-under-int'] },
        { tex: `= \\E\\big[${CHG('r\'(\\x_1)\\,J_K')}\\big]`, why: 'The chain rule through the end point.', uses: ['chain-rule'] },
        { tex: `J_{k+1} = ${CHG('\\big(1 + h\\,\\partial_x b_\\theta\\big)\\,J_k + h\\,\\partial_\\theta b_\\theta')}, \\quad J_0 = 0`, why: 'Differentiate one step in θ; the noise term has zero derivative.', uses: ['chain-rule'] },
      ],
    },
    after: ['The full gradient has length 1.66. With one sample per batch the score-function estimate has a relative error of 4.4 and the pathwise estimate 2.6; at 32 samples they reach 0.71 and 0.39, so the pathwise estimator needs roughly three times fewer samples here for the same accuracy. Truncation adds a bias that more samples cannot remove: the last 20 steps keep about half of the gradient (bias 0.53), the last 10 lose 83 percent of it, and the last step alone almost all. In this toy the mode a sample ends in is decided early, at high noise, and only the full gradient can move mass between modes. Truncation trades this bias for memory.'],
    check: {
      q: 'Why does the pathwise estimator need a differentiable reward while the score-function estimator does not?',
      opts: [
        ['The pathwise estimator multiplies by the gradient of the reward at the end point; the score-function estimator only evaluates the reward and differentiates the log-density of the path.', true],
        ['The score-function estimator approximates the gradient of the reward with finite differences between rollouts.', false],
        ['Both need it, but the score-function estimator can use a smoothed reward.', false],
      ],
      fb: { true: 'Right. This is why policy gradients are used with rewards from a vision–language model judge, an OCR check or a human label, and pathwise methods with differentiable reward networks.', false: 'Look at the two formulas: where does r appear in each, and is it differentiated?' },
    },
  },
  {
    id: 'dpo', kicker: 'Preference optimization', title: 'Preferences without a reward', scene: 'dpo',
    prose: [
      'Often there is no reward function, only pairs of samples where a person preferred one. The Bradley–Terry model says that the winner is preferred with probability \\(\\sigma(r(\\x^w) - r(\\x^l))\\) for some unknown reward. Direct preference optimization combines this with step 1: the reward can be written through the tilted target and the base model, so the likelihood of the labels becomes a loss on the model itself, with no reward model and no sampling during training.',
      'For a diffusion model the log-likelihood of a sample is not available. Diffusion-DPO replaces each log-ratio by the difference of the denoising losses of the trained and the reference model, which is the difference of their ELBOs up to a constant. The canvas trains both versions on the same pairs: the exact one, which uses the mixture density, and the ELBO one, which uses six seeded draws of noise and time per sample. Change the number of pairs and press Play.',
    ],
    tex: `\\mathcal{L}(${theta}) = -\\E\\log\\sigma\\Big(${beta}\\Big[\\log\\frac{${pth}(${xw})}{p_{\\mathrm{ref}}(${xw})} - \\log\\frac{${pth}(${xl})}{p_{\\mathrm{ref}}(${xl})}\\Big]\\Big)`,
    deriv: {
      have: ['The tilted target of step 1 with \\(p_{\\mathrm{ref}} = \\pdata\\): \\(p^* = p_{\\mathrm{ref}}\\,e^{r/\\beta}/Z\\).', 'The Bradley–Terry model \\(P(\\x^w \\succ \\x^l) = \\sigma\\big(r(\\x^w) - r(\\x^l)\\big)\\).'],
      goal: 'P(\\x^w \\succ \\x^l) = \\sigma\\Big(\\beta\\Big[\\log\\frac{p^*(\\x^w)}{p_{\\mathrm{ref}}(\\x^w)} - \\log\\frac{p^*(\\x^l)}{p_{\\mathrm{ref}}(\\x^l)}\\Big]\\Big)',
      plan: 'Solve the tilt for the reward and insert it into the Bradley–Terry model; the normalizer cancels in the difference.',
      title: 'The reward hidden in the optimal model', lines: [
        { tex: `\\log p^*(\\x) = \\log p_{\\mathrm{ref}}(\\x) + ${CHG('r(\\x)/\\beta - \\log Z')}`, why: 'The log of the tilted density.', uses: ['log-rules'] },
        { tex: `r(\\x) = ${CHG('\\beta\\log\\frac{p^*(\\x)}{p_{\\mathrm{ref}}(\\x)} + \\beta\\log Z')}`, why: 'Solve for the reward.', uses: ['log-rules'] },
        { tex: `r(\\x^w) - r(\\x^l) = \\beta\\log\\frac{p^*(\\x^w)}{p_{\\mathrm{ref}}(\\x^w)} - \\beta\\log\\frac{p^*(\\x^l)}{p_{\\mathrm{ref}}(\\x^l)} ${CHG('+ 0')}`, why: 'Insert the reward of both samples; β log Z appears twice with opposite signs.', uses: ['substitution'] },
      ],
    },
    after: ['Replacing \\(p^*\\) by \\(p_\\theta\\) and maximizing the likelihood of the observed labels gives the loss above; with infinitely many pairs its optimum is \\(p^*\\). With finite data it is not. With 1000 pairs the exact version ends at a KL of 0.039 to the target and the ELBO version at 0.144, from 2.84 for the base model. With 300 pairs they reach 0.065 and 0.201. With 100 pairs the exact version passes 0.224 and then climbs back to 0.49, because nothing in the loss constrains the model away from the pairs; the ELBO version stalls at about the same value. The ELBO surrogate adds noise and a bias from its fixed draws, which costs accuracy at every size here.'],
    check: {
      q: 'Why can DPO be trained without knowing the normalizer Z?',
      opts: [
        ['The loss only uses the difference of two rewards, and β log Z is the same for both samples of a pair.', true],
        ['For diffusion models Z is always one.', false],
        ['The ELBO surrogate estimates Z from the sampled noise.', false],
      ],
      fb: { true: 'Right. Preferences are comparisons, and comparisons are blind to a constant shift of the reward.', false: 'Look at the last line of the derivation: what happens to β log Z when two rewards are subtracted?' },
    },
  },
  {
    id: 'control', kicker: 'Stochastic optimal control', title: 'Why the noise must be memoryless', scene: 'control',
    prose: [
      'Fine-tuning can also be posed as control: keep the sampler of the base model and add a correction to its drift, penalized by its squared size. The optimal correction is the noise level times the gradient of \\(\\log h_t\\), where \\(h_t\\) is the expected exponentiated reward from the current point. The controlled path distribution is the base path distribution reweighted by \\(e^{r/\\beta}\\) and divided by \\(h_0\\) at the start. Since the start stays fixed, the end point is distributed as the target only if \\(h_0\\) is the same for every start.',
      'This holds when the end point of the base sampler is independent of its start, which is what the memoryless noise level \\(\\varepsilon_t^2 = 2(1-t)/t\\) of Adjoint Matching achieves on the linear path. With less noise, the start still decides part of the end, and the controlled sampler ends at a different density. The canvas solves for \\(h_t\\) and for the density of the controlled sampler on a grid. Move λ below one.',
    ],
    tex: `c^*_t = ${eps}\\,\\nabla\\log ${h}, \\qquad ${h}(${x}) = \\E\\big[e^{${r}(\\x_1)/${beta}}\\,\\big|\\,\\x_t = ${x}\\big], \\qquad q(\\x_1) = \\int p_0(\\x_0)\\,\\frac{p(\\x_1\\mid\\x_0)\\,e^{${r}(\\x_1)/${beta}}}{h_0(\\x_0)}\\,\\dd\\x_0`,
    deriv: {
      have: ['The controlled path density \\(p(\\tau)\\,e^{r(\\x_1)/\\beta}/h_0(\\x_0)\\) with the same start density \\(p_0\\) as the base sampler.', 'The base sampler is memoryless: \\(p(\\x_1\\mid\\x_0) = \\pdata(\\x_1)\\) for every \\(\\x_0\\).'],
      goal: 'q(\\x_1) = p^*(\\x_1)',
      plan: 'Compute \\(h_0\\) under the memoryless assumption, then integrate the start out of the controlled path density.',
      title: 'The controlled sampler ends at the target', lines: [
        { tex: `h_0(\\x_0) = \\int p(\\x_1\\mid\\x_0)\\,e^{r(\\x_1)/\\beta}\\,\\dd\\x_1 = ${CHG('\\int \\pdata(\\x_1)\\,e^{r(\\x_1)/\\beta}\\,\\dd\\x_1 = Z')}`, why: 'The definition of h at t = 0, and the memoryless assumption.', uses: ['expectation', 'substitution'] },
        { tex: `q(\\x_1) = \\int p_0(\\x_0)\\,\\frac{${CHG('\\pdata(\\x_1)')}\\,e^{r(\\x_1)/\\beta}}{${CHG('Z')}}\\,\\dd\\x_0`, why: 'Marginalize the controlled path density over the start, and insert both facts.', uses: ['substitution'] },
        { tex: `= \\frac{\\pdata(\\x_1)\\,e^{r(\\x_1)/\\beta}}{Z}${CHG('\\int p_0(\\x_0)\\,\\dd\\x_0')} = ${CHG('p^*(\\x_1)')}`, why: 'Nothing else depends on the start, and the start density integrates to one.', uses: ['fubini'] },
      ],
    },
    after: ['At the memoryless level the controlled density is at a Wasserstein distance of 0.007 from the target, and \\(\\log h_0\\) varies by 0.012 over the start, which is the error of starting at t = 0.01 rather than exactly at noise. At λ = 0.5 the variation is 0.29 and the distance 0.031; at λ = 0.25 they are 1.25 and 0.209, and at λ = 0.1, close to the ODE, 3.08 and 0.709. Policy gradients of step 4 are not affected, because they optimize the end distribution directly, but any method that learns the control from the base sampler must use memoryless noise during training. It may sample with any noise afterwards, since the fine-tuned drift defines a new model with its own ODE.'],
    check: {
      q: 'What is \\(\\varepsilon_t^2\\) at \\(t = 0.5\\) for the memoryless level, \\(\\lambda = 1\\)?',
      answer: 2, tol: 0.01,
      fb: { true: 'Right: \\(2(1 - 0.5)/0.5 = 2\\). The level is infinite at t = 0, which erases the start, and zero at t = 1, where the sample is fixed.', false: 'Insert t = 0.5 into \\(\\varepsilon_t^2 = 2(1 - t)/t\\).' },
    },
    closing: `<h3>Where this leads</h3>
      <p>All methods on this page aim at the same density, the base model tilted by the exponentiated reward. Policy gradients reach it with a stochastic sampler and the path likelihood, pathwise gradients with less variance but only for differentiable rewards, DPO from preference pairs without a reward model, and control methods with a memoryless sampler. None of them protects against a reward that is wrong outside its data; only the KL weight does.</p>
      <p>The tutorial <a href="../couplings/">Couplings, transport and control</a> (§2.6) derives the same control from path measures. The fine-tuned model is still a multi-step sampler; <a href="../distillation/">Distillation and few-step generation</a> (§5) shortens it, and <a href="../inference-time/">Inference-time scaling</a> (§6) reaches the tilted target at sampling time, without any training. Animations <a href="../../animations/reward-tilt/index.html">A7, reward tilting and hacking</a>, and <a href="../../animations/flow-grpo/index.html">A8, Flow-GRPO</a>, show the same ideas on a 2D toy.</p>`,
  },
];
