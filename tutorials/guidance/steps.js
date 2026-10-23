// Content of the §2.11 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex,
// section "Conditioning and guidance"; S(id) tags a symbol from symbols.js, CHG marks the part of a
// derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), c = S('c'), pt = S('pt'), ptc = S('ptc'), cls = S('cls'), s = S('s'), w = S('w'), sw = S('sw'), pi = S('pi');
const h = S('h'), ab = S('ab'), sweak = S('sweak'), N = S('N'), G = S('G');

export const INTRO = `
  <div class="kicker">Survey §2.11 · Interactive tutorial</div>
  <h1>Conditioning and guidance</h1>
  <p>A conditional diffusion model needs no new theory: the condition is one more input of the network, and the losses are the same. Sampled at face value, however, conditional models follow the condition only loosely, and nearly every deployed model strengthens it at sampling time with classifier-free guidance. This tutorial derives the conditional score from Bayes' rule, shows which density classifier-free guidance follows at each noise level, and then shows that the sampler does not end at that density, even with exact scores. It closes with three ways to do better: guiding only at some noise levels, guiding against a weaker model, and reweighting particles so that the intended target is sampled exactly.</p>
  <p>No step trains a network. Every canvas uses one exact distribution on the real line: five Gaussian modes at −3, −1.5, 0, 1.5 and 3 with width 0.3. Class A owns the two left modes, class B the two right ones, and the middle mode belongs to both with probability 1/2. Every noised density, the noisy classifier and every score have closed forms, and the samplers are integrated from 400 stratified quantiles of the noise, so the densities they produce are exact up to the step size. Random draws use fixed seeds, and the convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\).</p>
  <ol class="toc">
    <li><a href="#bayes">Bayes' rule on scores</a></li>
    <li><a href="#cfg">Classifier-free guidance and the density it follows</a></li>
    <li><a href="#jensen">CFG does not sample its target</a></li>
    <li><a href="#interval">Where to guide</a></li>
    <li><a href="#autoguidance">Autoguidance</a></li>
    <li><a href="#smc">Sampling the target with Feynman–Kac weights</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'bayes', kicker: 'Conditional scores', title: 'Bayes\' rule on scores', scene: 'bayes',
    prose: [
      'The conditional density of the noisy point is the unconditional density times the probability of the condition given that point, divided by the probability of the condition. The last factor does not depend on \\(\\x\\), so it disappears when the gradient of the log is taken. The conditional score is the unconditional score plus the gradient of the log of a classifier that sees noisy inputs. Classifier guidance trains such a classifier and adds its gradient, scaled by a weight w, to an unconditional model.',
      'The noisy classifier is not the classifier of clean data applied to a noisy point. It is the average of the clean classifier over all data points that could have produced \\(\\x_t\\). At high noise almost any data point could have, so the classifier is close to the prior 1/2 everywhere; at low noise it is 1 on the modes of A, 0 on the modes of B and 1/2 on the shared mode. Move t and the point x and check that the two scores on the right add up to the conditional score.',
    ],
    tex: `\\nabla\\log ${ptc} = \\nabla\\log ${pt} + \\nabla\\log ${cls}, \\qquad ${cls} = \\E\\big[p(${c}\\mid\\x_1)\\,\\big|\\,\\x_t = ${x}\\big]`,
    deriv: {
      have: ['The joint density of the noisy point and the condition, \\(p_t(\\x, c) = p_t(\\x)\\,p_t(c\\mid\\x) = p(c)\\,p_t(\\x\\mid c)\\).', 'The forward process: \\(\\x_t\\) is made from \\(\\x_1\\) and fresh noise, so c and \\(\\x_t\\) are independent given \\(\\x_1\\).'],
      goal: '\\nabla\\log p_t(\\x\\mid c) = \\nabla\\log p_t(\\x) + \\nabla\\log p_t(c\\mid\\x)',
      plan: 'Write Bayes\' rule, take the log, differentiate in x. Then compute the classifier by conditioning on the clean point.',
      title: 'The conditional score and the noisy classifier', lines: [
        { tex: `p_t(\\x\\mid c) = ${CHG('\\frac{p_t(\\x)\\,p_t(c\\mid\\x)}{p(c)}')}`, why: 'Bayes\' rule for the condition and the noisy point.', uses: ['bayes'] },
        { tex: `\\log p_t(\\x\\mid c) = \\log p_t(\\x) + \\log p_t(c\\mid\\x) ${CHG('- \\log p(c)')}`, why: 'The log of a product is the sum of the logs.', uses: ['log-rules'] },
        { tex: `\\nabla\\log p_t(\\x\\mid c) = \\nabla\\log p_t(\\x) + \\nabla\\log p_t(c\\mid\\x) ${CHG('+ 0')}`, why: 'The gradient is linear, and \\(\\log p(c)\\) does not depend on \\(\\x\\).', uses: ['grad-linear'] },
        { tex: `p_t(c\\mid\\x) = ${CHG('\\int p(c\\mid\\x_1, \\x_t = \\x)\\,p(\\x_1\\mid\\x_t = \\x)\\,\\dd\\x_1')}`, why: 'Condition on the clean point and average over its posterior given the noisy one.', uses: ['cond-exp'] },
        { tex: `= ${CHG('\\E\\big[p(c\\mid\\x_1)\\,\\big|\\,\\x_t = \\x\\big]')}`, why: 'Given \\(\\x_1\\), the noisy point carries no further information about c.', uses: ['expectation'] },
      ],
    },
    after: ['In this toy the classifier has a closed form: \\(p_t(A\\mid\\x) = \\sum_k p_t(k\\mid\\x)\\,p(A\\mid k)\\), the posterior of each mode times the share of A in that mode. Its log-gradient is largest between the shared mode and its neighbours, where a small move of \\(\\x\\) changes which mode the point most likely came from.'],
    check: {
      q: 'What is \\(p_0(A\\mid\\x)\\) at \\(t = 0\\), for any \\(\\x\\)?',
      answer: 0.5, tol: 0.01,
      fb: { true: 'Right. At t = 0 the noisy point is pure noise and independent of the data, so the posterior of \\(\\x_1\\) is the data distribution and the classifier is the prior \\(p(A) = 1/2\\). Its gradient is zero, and conditional and unconditional scores agree.', false: 'At t = 0 the point \\(\\x_0\\) carries no information about \\(\\x_1\\). Average \\(p(A\\mid\\x_1)\\) over the whole data distribution.' },
    },
  },
  {
    id: 'cfg', kicker: 'Classifier-free guidance', title: 'Classifier-free guidance and the density it follows', scene: 'cfg',
    prose: [
      'Classifier-free guidance avoids the classifier. One network is trained both with and without the condition, and at sampling time its two predictions are extrapolated with a scale w. The weights 1 − w and w sum to one, and the velocity, the noise prediction and the data prediction are all affine in the score with coefficients that do not depend on c, so the same extrapolation can be applied to any of them.',
      'By Bayes\' rule the guided score is the score of a density, \\(\\pi^w_t \\propto p_t(\\x)\\,p_t(c\\mid\\x)^w\\). Raising the classifier to the power w > 1 favours the points where the condition is unambiguous. At t = 1 the shared mode, where \\(p(A\\mid\\x) = 1/2\\), loses weight to the two modes that belong only to A. Move w and t; the dashed curve is the conditional density, which is \\(\\pi^1_t\\).',
    ],
    tex: `${sw} = (1 - ${w})\\,${s}(${x}) + ${w}\\,${s}(${x}\\mid ${c}) = \\nabla\\log ${pi}, \\qquad ${pi} \\propto ${pt}\\,${cls}^{${w}}`,
    deriv: {
      have: ['The guided score \\(\\s^w_t = (1 - w)\\nabla\\log p_t(\\x) + w\\nabla\\log p_t(\\x\\mid c)\\).', 'Bayes\' rule on scores from step 1.'],
      goal: '\\s^w_t = \\nabla\\log\\big(p_t(\\x)\\,p_t(c\\mid\\x)^w\\big)',
      plan: 'Replace the conditional score with step 1 and collect terms.',
      title: 'Which density CFG follows', lines: [
        { tex: `\\s^w_t = (1 - w)\\nabla\\log p_t(\\x) + w\\big(${CHG('\\nabla\\log p_t(\\x) + \\nabla\\log p_t(c\\mid\\x)')}\\big)`, why: 'Bayes\' rule on scores.', uses: ['bayes'] },
        { tex: `= ${CHG('\\nabla\\log p_t(\\x) + w\\nabla\\log p_t(c\\mid\\x)')}`, why: 'The two unconditional terms have weights that sum to one.', uses: ['grad-linear'] },
        { tex: `= \\nabla\\log\\big(${CHG('p_t(\\x)\\,p_t(c\\mid\\x)^w')}\\big)`, why: 'A multiple of a log is the log of a power, and a sum of logs is the log of the product.', uses: ['log-rules'] },
      ],
    },
    after: ['At t = 1 every mode is a separate bump, and the weight of mode k in \\(\\pi^w_1\\) is proportional to its weight in the data times \\(p(A\\mid k)^w\\). The modes of A keep factor 1, the shared mode gets \\((1/2)^w\\) and the modes of B get 0. At high noise the classifier is nearly flat, so \\(\\pi^w_t\\) stays close to the conditional density for every w.'],
    check: {
      q: 'All five modes have the same weight in the data. What share of \\(\\pi^w_1\\) lies on the shared mode at \\(w = 3\\)?',
      answer: 0.0588, tol: 0.004,
      fb: { true: 'Right. The weights are proportional to 1, 1 and \\((1/2)^3 = 1/8\\), so the shared mode has \\(\\tfrac{1/8}{2 + 1/8} = 1/17 \\approx 0.059\\), against 0.2 in the conditional density.', false: 'Multiply the weight of each mode by \\(p(A\\mid k)^3\\): 1 for the two modes of A, 1/8 for the shared mode, 0 for B. Then normalize.' },
    },
  },
  {
    id: 'jensen', kicker: 'What CFG samples', title: 'CFG does not sample its target', scene: 'jensen',
    prose: [
      'It is tempting to conclude that CFG samples \\(\\pi^w_1\\). It does not. Sampling \\(\\pi^w_1 \\propto p_1(\\x)\\,p(c\\mid\\x)^w\\) is a tilt of the data by the reward \\(w\\log p(c\\mid\\x)\\), and §2.6 showed that the exact tilt adds the score of \\(h_t(\\x) = \\E[p(c\\mid\\x_1)^w\\mid\\x_t = \\x]\\) to the unconditional score, the average of a power. CFG adds the score of \\(p_t(c\\mid\\x)^w\\), the power of an average. By Jensen\'s inequality the two differ for w > 1 wherever the clean point is uncertain, which is at high noise.',
      'The left plot runs both samplers from the same 400 noise quantiles with exact scores. The exact tilt lands on \\(\\pi^w_1\\). CFG decides too early: at high noise it already pushes toward the region where the classifier is most certain, and at large w it puts nearly every sample on the leftmost mode and beyond it, the extreme of the support of A. The right plot shows the two log factors at the chosen time.',
    ],
    tex: `${h}(${x}) = \\E\\big[p(${c}\\mid\\x_1)^{${w}}\\,\\big|\\,\\x_t = ${x}\\big] \\;\\ge\\; ${cls}^{${w}} \\quad\\text{for } ${w} \\ge 1`,
    deriv: {
      have: ['The definitions \\(h_t = \\E[p(c\\mid\\x_1)^w\\mid\\x_t]\\) and \\(p_t(c\\mid\\x) = \\E[p(c\\mid\\x_1)\\mid\\x_t]\\).', 'The function \\(y \\mapsto y^w\\) is convex on [0, 1] for w ≥ 1.'],
      goal: 'h_t(\\x) \\ge p_t(c\\mid\\x)^w',
      plan: 'Apply Jensen\'s inequality to the posterior expectation.',
      title: 'Power of an average against average of a power', lines: [
        { tex: `h_t(\\x) = \\E\\big[${CHG('\\varphi(p(c\\mid\\x_1))')}\\,\\big|\\,\\x_t = \\x\\big], \\quad \\varphi(y) = y^w`, why: 'Name the convex function.', uses: ['expectation'] },
        { tex: `\\ge ${CHG('\\varphi\\big(\\E[p(c\\mid\\x_1)\\mid\\x_t = \\x]\\big)')}`, why: 'Jensen\'s inequality for a convex function. Equality holds when \\(p(c\\mid\\x_1)\\) is the same for all likely \\(\\x_1\\).', uses: ['jensen'] },
        { tex: `= ${CHG('p_t(c\\mid\\x)^w')}`, why: 'Step 1.', uses: ['cond-exp'] },
      ],
    },
    after: ['There is a second, separate problem. The densities \\(\\pi^w_t\\) of step 2 are not the marginals of any noising process, because powers do not commute with Gaussian convolution, so even a sampler that followed \\(\\s^w\\) perfectly would not keep its samples distributed as \\(\\pi^w_t\\). At t = 0 the ratio of the two factors is largest: \\(h_0\\) is the average of \\(p(A\\mid\\x_1)^w\\) over all data, while \\(p_0(A\\mid\\x)^w = (1/2)^w\\).'],
    check: {
      q: 'At \\(t = 0\\) and \\(w = 3\\), take \\(p(A\\mid\\x_1)\\) to be 1, 1, 1/2, 0 and 0 on the five equally likely modes. What is the ratio \\(h_0 / p_0(A\\mid\\x)^3\\)?',
      answer: 3.4, tol: 0.05,
      fb: { true: 'Right. \\(h_0 = \\tfrac15(1 + 1 + \\tfrac18) = 0.425\\) and \\(p_0(A\\mid\\x)^3 = (1/2)^3 = 0.125\\), a ratio of 3.4. The exact tilt is flat at t = 0, so it does not matter for the score, but at intermediate t the two factors have different shapes, and the shapes set the scores.', false: 'At t = 0 the posterior of \\(\\x_1\\) is the data. Average \\(p(A\\mid k)^3\\) over the five modes for \\(h_0\\), and cube the average of \\(p(A\\mid k)\\) for the other factor.' },
    },
  },
  {
    id: 'interval', kicker: 'Practical variants', title: 'Where to guide', scene: 'interval',
    prose: [
      'A trained model is not exact. From here on the conditional model has modes that are 0.45 wide instead of 0.3, as an underfitted network has, and it ignores the condition for 15% of its mass, so some of its samples land on the modes of B. The unconditional model is the same network with the condition dropped. Precision is the share of samples within 0.6 of a mode of A. Coverage adds, over the five modes, the smaller of the generated share and the share in \\(p(\\x\\mid A)\\), so it is 1 only when every mode of A gets its share.',
      'Guidance at different times decides different things. At high noise, small t, the paths have not chosen a mode yet, and guidance pushes them away from B and away from the shared mode, which costs coverage. At low noise every path has chosen its mode, the noisy classifier is constant within each mode, and guidance changes almost nothing. Restricting guidance to an interval of intermediate times keeps most of the coverage of the conditional model. Move the interval and watch the paths and the shares of the five modes.',
    ],
    tex: `\\s^{w}_t \\text{ for } t \\in ${ab}, \\qquad \\s_t(\\x\\mid ${c}) \\text{ otherwise}`,
    after: ['With w = 3 on all of [0, 1], the shared mode drops from a share of 0.20 to almost nothing and the leftmost mode receives 0.83. With guidance only on [0.5, 1] the shares stay close to those of the conditional model and the distance to \\(p(\\x\\mid A)\\) is smaller than without guidance. The guidance interval of Kynkäänniemi et al. is this restriction, chosen on image benchmarks.'],
    check: {
      q: 'At \\(t = 0.9\\) the noisy classifier is nearly constant within each mode. Inside one mode, by how much does the CFG score differ from the conditional score?',
      answer: 0, tol: 0.01,
      fb: { true: 'Right. The difference is \\((w - 1)\\nabla\\log p_t(c\\mid\\x)\\), which vanishes where the classifier is flat. Late guidance can only act between the modes.', false: 'Write \\(\\s^w_t - \\s_t(\\x\\mid c) = (w - 1)(\\s_t(\\x\\mid c) - \\s_t(\\x)) = (w - 1)\\nabla\\log p_t(c\\mid\\x)\\) and use that the classifier is flat.' },
    },
  },
  {
    id: 'autoguidance', kicker: 'Practical variants', title: 'Autoguidance', scene: 'autoguidance',
    prose: [
      'CFG extrapolates away from the unconditional model, so it changes two things at once: it strengthens the condition and it moves away from the errors of the model. Autoguidance keeps only the second. It extrapolates away from a weaker version of the conditional model itself, smaller or trained for less time. The weak model makes the same kind of errors as the good one, only larger, so the difference of the two points away from those errors and leaves the condition alone.',
      'Here the weak model has modes 0.65 wide where the good model has 0.45. The difference of their scores points toward the centre of every mode, the shared one included, so autoguidance sharpens all modes at once. Move w and compare the generated densities and the two curves on the right.',
    ],
    tex: `${sweak} + ${w}\\,\\big(${s}(${x}\\mid ${c}) - ${sweak}\\big)`,
    after: ['At w = 4, CFG has precision 0.81 and coverage 0.43, since nearly every sample is on the leftmost mode. Autoguidance has precision 0.92 and coverage 0.96: the modes are sharper and keep their shares. It does not remove the small leaked mass on the modes of B (a share of 0.04), because the weak model puts mass there too. Only an extrapolation away from a model without the condition removes it.'],
    check: {
      q: 'For which value of \\(w\\) does autoguidance return the good model, whatever the weak model is?',
      answer: 1, tol: 0.01,
      fb: { true: 'Right. At \\(w = 1\\) the weak model cancels. For \\(w > 1\\) the extrapolation goes beyond the good model, away from the weak one.', false: 'Expand \\(\\s_{\\mathrm{weak}} + w(\\s - \\s_{\\mathrm{weak}}) = (1 - w)\\s_{\\mathrm{weak}} + w\\s\\) and find when the first weight is zero.' },
    },
  },
  {
    id: 'smc', kicker: 'Sampling the tilted target', title: 'Sampling the target with Feynman–Kac weights', scene: 'smc',
    prose: [
      'Step 3 showed the exact tilt \\(h_t\\), but for a real model it is an expectation over the posterior of the clean image that nobody can compute. Feynman–Kac correctors avoid it. Several particles run the conditional sampler together; after every step each particle is reweighted by the change of a potential \\(G_t = p_t(c\\mid\\x)^{w - 1}\\), which a classifier-free model provides, and when the weights become too uneven the particles are resampled in proportion to their weights.',
      'The product of the weight changes along a path telescopes to the potential at the end, \\(p(c\\mid\\x_1)^{w - 1}\\), so the weighted particles target \\(p(\\x\\mid c)\\,p(c\\mid\\x)^{w - 1} \\propto \\pi^w_1\\), whatever the intermediate potentials are. They only decide when the sampler resamples. Here each run has N particles on the exact conditional SDE and returns one particle drawn by its weight, and the plot uses 400 runs. Increase N and watch the distance to \\(\\pi^w_1\\) fall to the floor of 400 exact draws.',
    ],
    tex: `${G}(${x}) = ${cls}^{${w} - 1}, \\qquad \\prod_{k} \\frac{G_{t_{k+1}}(\\x_{t_{k+1}})}{G_{t_k}(\\x_{t_k})} = \\frac{p(${c}\\mid\\x_1)^{${w} - 1}}{G_{t_0}(\\x_{t_0})}`,
    deriv: {
      have: ['Paths of the conditional sampler, whose end point has density \\(p(\\x_1\\mid c)\\).', 'A start weight \\(G_{t_0}(\\x_{t_0})\\) and the weight changes \\(G_{t_{k+1}}/G_{t_k}\\) along the path.'],
      goal: 'p(\\x_1\\mid c)\\,p(c\\mid\\x_1)^{w-1} \\propto \\pi^w_1(\\x_1)',
      plan: 'Multiply the weights, cancel the intermediate potentials, and apply Bayes\' rule to the result.',
      title: 'Why the weights hit the target', lines: [
        { tex: `\\text{weight} = G_{t_0}(\\x_{t_0})\\prod_k \\frac{G_{t_{k+1}}(\\x_{t_{k+1}})}{G_{t_k}(\\x_{t_k})} = ${CHG('G_1(\\x_1) = p(c\\mid\\x_1)^{w - 1}')}`, why: 'Each potential appears once in a numerator and once in the next denominator.', uses: ['log-rules'] },
        { tex: `p(\\x_1\\mid c)\\,p(c\\mid\\x_1)^{w - 1} = ${CHG('\\frac{p(\\x_1)\\,p(c\\mid\\x_1)}{p(c)}')}\\,p(c\\mid\\x_1)^{w - 1}`, why: 'Bayes\' rule for the conditional density of the end point.', uses: ['bayes'] },
        { tex: `\\propto ${CHG('p(\\x_1)\\,p(c\\mid\\x_1)^w')} = \\pi^w_1`, why: 'p(c) is a constant, and the powers add.', uses: ['log-rules'] },
      ],
    },
    after: ['With one particle no reweighting can happen, and the sampler draws from the conditional density, a distance of 0.28 from \\(\\pi^3_1\\). Two particles cut it to less than a third, and from about eight particles on the distance is that of 400 exact draws. CFG, with one model evaluation per step and no particles, stays at 0.86. The price is N samplers per output and a classifier-free model evaluated for every particle. §2.6 derives the same weights as importance weights of path measures, and the chapter on inference-time scaling treats SMC samplers for general rewards.'],
    check: {
      q: 'With \\(N = 1\\), what share of the outputs should land on the shared mode?',
      answer: 0.2, tol: 0.02,
      fb: { true: 'Right. A single particle is never compared with another, so the output is a plain conditional sample, and the shared mode has weight 0.2 in \\(p(\\x\\mid A)\\). The run here gives 0.185 from 400 samples.', false: 'With one particle the weights cannot change which particle is returned. What is the weight of the shared mode in \\(p(\\x\\mid A)\\)?' },
    },
    closing: `<h3>Where this leads</h3>
      <p>Conditioning is Bayes' rule on scores, and classifier-free guidance is a cheap way to raise the noisy classifier to a power. The sampler that results does not sample the sharpened density that the power suggests, because the power of an average is not the average of a power, and in this toy it ends far from it. Guidance intervals and autoguidance repair parts of the damage on a trained model; Feynman–Kac weights sample the intended density exactly, at the price of many particles.</p>
      <p>The earlier tutorials <a href="../couplings/">Couplings, transport and control</a>, <a href="../sde/">Diffusion as an SDE</a> and <a href="../hybrid/">Autoregressive and hybrid visual generation</a> develop the pieces used on this page.</p>`,
  },
];
