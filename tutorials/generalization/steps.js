// Content of the §2.7 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex,
// section "Why diffusion models generalize"; S(id) tags a symbol from symbols.js, CHG marks the part of
// a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), t = S('t'), y = S('y'), n = S('n'), w = S('w'), hx = S('hx1'), u = S('vel'), al = S('alpha'), sg = S('sig');
const ph = S('phat'), x0 = S('x0'), Phi = S('Phi'), h = S('h'), P = S('P'), s = S('s');
const yi = S('y', '\\y_i'), wi = S('w', 'w_i');

export const INTRO = `
  <div class="kicker">Survey §2.7 · Interactive tutorial</div>
  <h1>Why diffusion models generalize</h1>
  <p>Every training objective of this chapter is a regression with a known minimizer. Training uses a finite set of examples, so the exact minimizer is the field of the empirical distribution, the uniform distribution on the training set. This tutorial computes that field exactly and shows that it only reproduces the training data. It then follows the questions the survey asks next: at which time along the path a sample commits to a class and then to a single training point, how two models trained on different data can agree, why some training points are memorized while others are not, and how two constraints of convolutional networks, locality and translation equivariance, produce new images without any training. Each step states one idea, shows one formula and gives you a plot that the formula controls. Every symbol can be hovered or tapped, and the button "Symbols" in the top bar opens the full list.</p>
  <p>The convention is that of the survey: \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), where \\(t = 0\\) is noise and \\(t = 1\\) is data, on the linear path \\(\\alpha_t = t\\), \\(\\sigma_t = 1 - t\\). The data are the 2D Gaussian mixture of the earlier tutorials. In higher dimensions the extra coordinates are independent Gaussian noise of standard deviation 0.5. The empirical field is a Gaussian mixture with one component per training point, so it is computed exactly. Two stand-ins replace a trained network, because a real network would have to be trained: a smoothed empirical distribution for a model that cannot resolve fine detail, and a Gaussian fitted to the data for a linear model. Step 6 uses the patch-based closed form of Kamb and Ganguli on small images of digits. All samples come from integrating the probability-flow ODE with fixed noise.</p>
  <ol class="toc">
    <li><a href="#empirical">The optimum of the training loss</a></li>
    <li><a href="#copies">The exact optimum copies</a></li>
    <li><a href="#collapse">Speciation, then collapse</a></li>
    <li><a href="#agree">Two models on disjoint data</a></li>
    <li><a href="#local">Memorization is local</a></li>
    <li><a href="#patches">Locality and equivariance create new images</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'empirical', kicker: 'What training minimizes', title: 'The optimum of the training loss', scene: 'empirical',
    prose: [
      'Denoising score matching and flow matching regress onto \\(\\x_1\\) given \\(\\x_t\\). Training draws \\(\\x_1\\) from the training set \\(\\{\\y_1, \\dots, \\y_n\\}\\), so the loss that is minimized is that of the empirical distribution \\(\\hat p = \\frac1n\\sum_i\\delta_{\\y_i}\\). Its minimizer is known in closed form. Noising \\(\\hat p\\) gives a Gaussian mixture with one component \\(\\N(\\alpha_t\\y_i, \\sigma_t^2 I)\\) per training point, and the optimal denoiser is the weighted mean of the training points in the formula.',
      'The weight \\(w_i\\) is the posterior probability that \\(\\x_t\\) was made from \\(\\y_i\\). Drag the ring \\(\\x_t\\) in the plot. The halos show the weights, the dashed circle of radius \\(\\sigma_t/\\alpha_t\\) around \\(\\x_t/\\alpha_t\\) marks the points that share most of the weight, and the arrow points to \\(\\hat\\x_1\\). Move \\(t\\) towards 1. The circle shrinks, one weight takes over, and \\(\\hat\\x_1\\) jumps to a single training point. The bars on the right are the largest weights in decreasing order.',
    ],
    tex: `${hx}(\\x, ${t}) = \\sum_{i=1}^{${n}} ${wi}(\\x, ${t})\\,${yi}, \\qquad ${wi}(\\x, ${t}) = \\operatorname{softmax}_i\\Big(-\\frac{\\|\\x - ${al}\\,${yi}\\|^2}{2${sg}^2}\\Big)`,
    deriv: {
      have: ['\\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) with \\(\\x_0\\sim\\N(0, I)\\) and \\(\\x_1\\) uniform on \\(\\{\\y_1, \\dots, \\y_n\\}\\).', 'The regression lemma of §2.2: the minimizer of \\(\\E\\|\\hat\\x_1(\\x_t) - \\x_1\\|^2\\) is \\(\\E[\\x_1\\mid\\x_t]\\).'],
      goal: `${hx}(\\x, ${t}) = \\sum_i ${wi}\\,${yi}`,
      plan: 'Write the conditional expectation as a sum over the training points and compute each posterior probability with Bayes’ rule.',
      title: 'Why the optimum is a softmax-weighted mean', lines: [
        { tex: `${hx}(\\x, ${t}) = \\E[\\x_1\\mid ${x} = \\x] = \\sum_{i=1}^{${n}} ${yi}\\;P(\\x_1 = ${yi}\\mid ${x} = \\x)`, why: 'The minimizer of the squared loss is the conditional mean, and \\(\\x_1\\) takes only the n values \\(\\y_i\\).', uses: ['regression-lemma', 'expectation'] },
        { tex: `P(\\x_1 = ${yi}\\mid ${x} = \\x) = \\frac{${CHG('\\tfrac1n\\,\\N(\\x;\\ \\alpha_t\\y_i,\\ \\sigma_t^2 I)')}}{\\sum_j \\tfrac1n\\,\\N(\\x;\\ \\alpha_t\\y_j,\\ \\sigma_t^2 I)}`, why: 'Bayes’ rule. The prior of each training point is 1/n, and given \\(\\x_1 = \\y_i\\) the point \\(\\x_t\\) is Gaussian with mean \\(\\alpha_t\\y_i\\) and covariance \\(\\sigma_t^2 I\\).', uses: ['bayes', 'gauss'] },
        { tex: `= \\frac{\\exp\\big(-\\|\\x - \\alpha_t ${yi}\\|^2/2\\sigma_t^2\\big)}{\\sum_j \\exp\\big(-\\|\\x - \\alpha_t\\y_j\\|^2/2\\sigma_t^2\\big)} = ${CHG(wi)}`, why: 'The factor 1/n and the normalizing constant of the Gaussian are the same for every i and cancel. What remains is a softmax of the negative squared distances.', uses: ['log-rules'] },
      ],
    },
    after: ['Nothing in the formula refers to \\(\\pdata\\). At \\(t = 0\\) the weights are uniform and \\(\\hat\\x_1\\) is the mean of the training set. As \\(\\sigma_t/\\alpha_t\\) falls below the typical distance between neighbouring training points, the softmax selects the nearest one. A network that minimized the training loss exactly would be this function.'],
    check: {
      q: 'What does the empirical denoiser return at \\(t = 0\\), whatever the input?',
      opts: [['The mean of the training set', true], ['The training point nearest to the input', false], ['The input itself', false]],
      fb: { true: 'Right. At \\(t = 0\\), \\(\\alpha_t = 0\\), so every distance \\(\\|\\x - \\alpha_t\\y_i\\|\\) is the same, the weights are all 1/n and \\(\\hat\\x_1\\) is the average of the training points. Set t to 0 and drag the ring: the star does not move.', false: 'At \\(t = 0\\) the signal coefficient \\(\\alpha_t\\) is zero, so all squared distances \\(\\|\\x - \\alpha_t\\y_i\\|^2\\) equal \\(\\|\\x\\|^2\\). The softmax of equal numbers is uniform, and \\(\\hat\\x_1\\) is the mean of the training set.' },
    },
  },
  {
    id: 'copies', kicker: 'Sampling with the exact optimum', title: 'The exact optimum copies', scene: 'copies',
    prose: [
      'On the linear path the velocity is a function of the denoiser, the formula below. Integrating the probability-flow ODE with the empirical denoiser therefore samples from the minimizer of the training loss. The left panel does this for 200 noise draws. The right panel integrates the same noise draws with the exact field of \\(\\pdata\\), the target that training is meant to approximate.',
      'Press Play or move \\(t\\). On the left every path ends on a training point. The readout applies the copy test of Gu et al. (2025): a sample is a copy when its distance to the nearest training point is less than a third of the distance to the second nearest. Move \\(n\\). Larger training sets give more distinct end points, but the share of copies stays at one. The samples of \\(\\pdata\\) on the right pass the same test only by chance, about as often as a fresh draw does.',
    ],
    tex: `${u}(\\x) = \\frac{${hx}(\\x, ${t}) - \\x}{1 - ${t}}, \\qquad \\dot{${S('X')}} = ${u}(${S('X')}), \\quad ${S('X', '\\bm X_0')} = ${x0}`,
    deriv: {
      have: ['\\(\\x_t = t\\x_1 + (1 - t)\\x_0\\), so the conditional velocity is \\(\\x_1 - \\x_0\\).', 'The marginal velocity of §2.4, \\(\\bm u_t(\\x) = \\E[\\x_1 - \\x_0\\mid\\x_t = \\x]\\).'],
      goal: `${u}(\\x) = \\frac{${hx} - \\x}{1 - t}`,
      plan: 'Eliminate \\(\\x_0\\) using the interpolation, then take the conditional expectation.',
      title: 'Why the velocity is a function of the denoiser', lines: [
        { tex: `\\x_0 = \\frac{${x} - t\\,\\x_1}{1 - t} \\quad\\Rightarrow\\quad \\x_1 - \\x_0 = ${CHG('\\frac{\\x_1 - \\x_t}{1 - t}')}`, why: 'Solve the interpolation for the noise and substitute it into the conditional velocity.', uses: ['substitution'] },
        { tex: `${u}(\\x) = \\E\\Big[\\frac{\\x_1 - \\x_t}{1 - t}\\,\\Big|\\,\\x_t = \\x\\Big] = \\frac{${CHG('\\E[\\x_1\\mid\\x_t = \\x]')} - \\x}{1 - t}`, why: 'Given \\(\\x_t = \\x\\), the term \\(\\x_t\\) is the constant \\(\\x\\), and the conditional expectation is linear.', uses: ['cond-exp'] },
        { tex: `= \\frac{${hx}(\\x, t) - \\x}{1 - t}`, why: 'The conditional mean of \\(\\x_1\\) is the denoiser. With the empirical denoiser of step 1, near \\(t = 1\\) it equals the nearest training point, and the velocity pulls the path onto it.', uses: ['regression-lemma'] },
      ],
    },
    after: ['A model that reaches the minimum of its training loss reproduces its training set and nothing else. Trained networks do produce new samples, so generalization has to come from what the network does not fit: its limited capacity, its architecture and the implicit bias of training. The next steps look at when the copying happens along the path and at three ways of not fitting the empirical field.'],
    check: {
      q: 'You multiply the size of the training set by 64. What happens to the samples of the exact empirical optimum?',
      opts: [['They are still copies; they land on more distinct training points', true], ['They become new samples once n is large enough', false], ['They converge to the mean of the data', false]],
      fb: { true: 'Right. For every finite n the empirical denoiser selects one training point as \\(t\\to1\\), so every path ends on a training point. A larger n spreads the samples over more points and makes the set of samples look more like \\(\\pdata\\), but each sample is still a copy. Move n from 8 to 512 and watch the readout.', false: 'For every finite n the weights concentrate on the nearest training point as \\(\\sigma_t\\to0\\), so every path ends on one of the \\(\\y_i\\). More data only means more distinct targets. Move n and watch the share of copies stay at one.' },
    },
  },
  {
    id: 'collapse', kicker: 'When along the path', title: 'Speciation, then collapse', scene: 'collapse',
    prose: [
      'Biroli and coauthors studied the exact empirical field in high dimension and found two transitions along the sampling path. At the speciation time \\(t_S\\) a path commits to a class: the total weight of the training points of one class approaches one. At the later collapse time \\(t_C\\) it commits to a single training point: the largest weight approaches one. Between the two the path is in the right class but not yet on a training point, and this is the part of the path that a model can use to generalize.',
      'The left panel shows 100 paths of the empirical field, coloured by the class they end in. The right panel plots the two quantities of the formula over \\(t\\), each averaged over the paths, and marks the times at which they reach 0.9. Move \\(n\\): collapse moves later as the training points become denser, while speciation hardly moves. Move \\(d\\): in higher dimension collapse comes earlier for the same \\(n\\), which is the curse of dimensionality for an exactly trained model.',
    ],
    tex: `P(${S('k')}\\mid ${x}) = \\sum_{i:\\,k_i = ${S('k')}} ${wi}(${x}, ${t}) \\quad\\text{reaches 1 at } ${S('tS')}, \\qquad ${S('wmax')}(${x}, ${t}) \\quad\\text{reaches 1 at } ${S('tC')}`,
    after: ['In the theory of Biroli et al. the speciation time is set by the spectrum of the data covariance, which describes how far apart the classes are, and the collapse time by an excess entropy, which compares the volume of the data with the number of training points. Doubling n adds one bit, while the volume of the data grows exponentially with d. Here the classes are the same in every dimension, so \\(t_S\\) stays near 0.6, while \\(t_C\\) moves down by a lot as d grows. Estimates of both times from 100 paths are noisy for the smallest training sets.'],
    check: {
      q: 'Fix \\(n = 512\\) and raise \\(d\\) from 2 to 16. What happens to the collapse time?',
      opts: [['It moves earlier, so more of the path is spent on a single training point', true], ['It moves later, because extra noise coordinates hide the training points', false], ['It does not change, because the extra coordinates carry no structure', false]],
      fb: { true: 'Right. The squared distance between two training points grows with d, because every independent coordinate adds to it. The softmax separates the training points once \\(\\sigma_t^2\\) is small compared with these differences, which happens at larger \\(\\sigma_t\\). The readout shows \\(t_C\\) falling from about 0.98 to about 0.73.', false: 'Look at the second vertical line as you move d. The extra coordinates carry no class information, but they increase the distances between training points. Larger gaps let the softmax pick out the nearest point at a higher noise level, so collapse comes earlier.' },
    },
  },
  {
    id: 'agree', kicker: 'A test for generalization', title: 'Two models on disjoint data', scene: 'agree',
    prose: [
      'Kadkhodaie et al. trained two networks on disjoint halves of an image dataset and started both from the same noise. With small halves each network reproduced its own training images. With large halves the two networks produced nearly the same image, so they had learned nearly the same field, and that field could not be the empirical one of either half. The plot repeats the test with the sets A and B of size \\(n\\) each and 64 shared noise draws. Each grey segment joins the sample of model A to the sample of model B from the same \\(\\x_0\\).',
      'Three models are compared. The empirical optimum of step 2, the smoothed empirical distribution with bandwidth 0.3 of step 5, and a Gaussian fitted to the training set. The Gaussian is a linear model, and its flow map is the affine map of the formula. Choose a model and move \\(n\\) and \\(d\\). The bars on the right give, for all three models, the mean disagreement, the share of copies of set A, and how well the samples fit \\(\\pdata\\).',
    ],
    tex: `\\text{disagreement} = \\E_{${x0}}\\big\\|${Phi}^{A}(${x0}) - ${Phi}^{B}(${x0})\\big\\|, \\qquad \\text{Gaussian model: } ${Phi}(${x0}) = ${S('mu')} + ${S('Sig')}^{1/2}\\,${x0}`,
    after: ['The empirical models agree in two dimensions once n is large, because the training points of A and B are then close to each other. In sixteen dimensions they disagree for every n on the slider, because the nearest training points of the two sets stay far apart. The Gaussian models agree with each other already for a moderate n in any dimension, since the mean and covariance need only about d samples to estimate. Their samples are not copies, but they also miss the three modes: the mean log density of their samples under \\(\\pdata\\) is far below that of real data. Maillard and Goldt compute this transition exactly for linear models and point out the same gap: agreement between two models measures one aspect of generalization, and recovering the structure of the data is another.'],
    check: {
      q: 'At \\(d = 16\\) and \\(n = 512\\), which statement fits the readout?',
      opts: [['The Gaussian models agree with each other but do not reproduce the modes of the data', true], ['The empirical models agree, so they generalize', false], ['All three models disagree, so none of them generalizes', false]],
      fb: { true: 'Right. The two Gaussian models differ by a small fraction of the distance between two random data points, and none of their samples is a copy. Their mean log density under \\(\\pdata\\) is still much lower than that of real samples, because a Gaussian puts mass between the modes. Agreement is necessary for generalization but not sufficient.', false: 'Select the Gaussian model at d = 16 and n = 512. Its two flow maps nearly agree, but the mean log density of its samples is far below the value for real data. The empirical models disagree at this size, and they copy.' },
    },
  },
  {
    id: 'local', kicker: 'Which points are copied', title: 'Memorization is local', scene: 'local',
    prose: [
      'A network cannot resolve arbitrarily small details, so it does not fit the empirical field exactly. A simple stand-in for this limit is to smooth the empirical distribution with a Gaussian of width \\(h\\), the kernel density estimate in the formula. Its denoiser is again a softmax over training points, with wider components, as the derivation below shows. For \\(h = 0\\) it is the empirical optimum.',
      'The training set has 160 points from the mixture, five isolated points far from the modes and one point that appears ten times. The plot shows 400 samples from the same noise; filled samples are copies, open ones are not. Move \\(h\\). In the dense regions copying stops at a small bandwidth, because neighbouring bumps merge into one smooth density. The isolated points and the duplicated point are still copied when the bulk is not. Merger and coauthors make this precise through the connection to kernel density estimation: a point is memorized when its neighbourhood is sparsely covered by the rest of the data.',
    ],
    tex: `${S('phath')} = \\frac{1}{${n}}\\sum_{i=1}^{${n}}\\N(${yi},\\ ${h}^2 I), \\qquad ${hx}(\\x, ${t}) = \\sum_i ${wi}\\Big(${yi} + \\frac{${al}${h}^2}{${al}^2${h}^2 + ${sg}^2}\\big(\\x - ${al}${yi}\\big)\\Big)`,
    deriv: {
      have: ['\\(\\x_1\\sim\\hat p^{\\,h}\\): pick i uniformly, then \\(\\x_1 = \\y_i + h\\bm\\varepsilon\\) with \\(\\bm\\varepsilon\\sim\\N(0, I)\\).', '\\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\).'],
      goal: `${hx} = \\sum_i ${wi}\\,\\E[\\x_1\\mid \\x_t, i]`,
      plan: 'Condition on the component i first. Within one component everything is Gaussian, so the posterior mean is linear in \\(\\x_t\\).',
      title: 'Denoiser of the smoothed empirical distribution', lines: [
        { tex: `\\x_t\\mid i \\;\\sim\\; \\N\\big(\\alpha_t ${yi},\\ ${CHG('v_t')}\\,I\\big), \\qquad v_t = \\alpha_t^2 ${h}^2 + \\sigma_t^2`, why: 'Given i, \\(\\x_t = \\alpha_t\\y_i + \\alpha_t h\\bm\\varepsilon + \\sigma_t\\x_0\\) is a sum of independent Gaussians, whose variances add.', uses: ['gauss'] },
        { tex: `${wi} = P(i\\mid\\x_t = \\x) = \\operatorname{softmax}_i\\Big(-\\frac{\\|\\x - \\alpha_t ${yi}\\|^2}{2${CHG('v_t')}}\\Big)`, why: 'Bayes’ rule as in step 1, with the wider variance \\(v_t\\) in place of \\(\\sigma_t^2\\).', uses: ['bayes'] },
        { tex: `\\E[\\x_1\\mid\\x_t = \\x, i] = ${yi} + \\frac{\\alpha_t ${h}^2}{v_t}\\big(\\x - \\alpha_t ${yi}\\big)`, why: 'Posterior mean of jointly Gaussian variables: the prior mean plus the covariance \\(\\alpha_t h^2 I\\) of \\(\\x_1\\) and \\(\\x_t\\), divided by the variance \\(v_t\\) of \\(\\x_t\\), times the residual.', uses: ['gauss', 'cond-exp'] },
        { tex: `${hx}(\\x, t) = \\sum_i ${wi}\\,\\E[\\x_1\\mid\\x_t = \\x, i]`, why: 'Average over the components with their posterior probabilities.', uses: ['cond-exp', 'expectation'] },
      ],
    },
    after: ['The survey draws two consequences. Audits for memorization should look at rare and duplicated training examples rather than at averages over the dataset. And anything that sharpens the effective target, such as strong conditioning, strong guidance or long fine-tuning on a small dataset, removes the smoothing and moves a model back towards the empirical optimum of step 2. Extraction attacks on large text-to-image models find exactly such cases: Carlini et al. recover training images, many of them duplicated in the training data, and Somepalli et al. find copied content in generated images.'],
    check: {
      q: 'Set \\(h = 0.2\\). Which samples are still mostly copies?',
      opts: [['Those near the isolated points and the duplicated point', true], ['Those in the densest part of the modes', false], ['None; at this bandwidth nothing is copied', false]],
      fb: { true: 'Right. In the modes the bumps of neighbouring points overlap, the smoothed density has no peak at any single point and only a small share of samples pass the copy test, about as many as for fresh draws of the data. An isolated point keeps its own bump, which the ODE still finds, and the duplicated point carries ten times the weight of a single one.', false: 'Read the readout at h = 0.2. The share of copies in the bulk is small, while most samples near the isolated points and the duplicated point are copies. Memorization depends on how densely the neighbourhood of a point is covered.' },
    },
  },
  {
    id: 'patches', kicker: 'Creativity from constraints', title: 'Locality and equivariance create new images', scene: 'patches',
    prose: [
      'Convolutional denoisers have two built-in constraints. Each output pixel depends only on a neighbourhood of the input, its receptive field, and the network treats every position in the same way, which is translation equivariance. Kamb and Ganguli derived the optimal denoiser under exactly these constraints and found that it explains much of what trained convolutional models produce. Each pixel of \\(\\hat\\x_1\\) becomes a weighted average of training pixels, where the weights compare the patch of \\(\\x_t\\) around that pixel with patches of the training images, the formula.',
      'The training set consists of the eight digits in the top row. The six columns below are samples, integrated in 30 steps from fixed noise. The upper image of each column is \\(\\x_t\\) and the lower one is the current \\(\\hat\\x_1\\). Move \\(P\\) and \\(s\\), and the run restarts. With the whole image as the patch and \\(s = 0\\), this is the empirical optimum of step 1, and each sample is a copy. With patches of 5 pixels and a small shift window, different regions of one sample match patches of different digits, and the samples become new digit-like shapes that are not in the training set.',
    ],
    tex: `${hx}(\\x, ${t})[p] = \\sum_{i}\\sum_{|q - p|\\le ${s}} W_{i,q}\\,${S('y', '\\y_i')}[q], \\qquad W_{i,q} = \\operatorname{softmax}_{i,q}\\Big(-\\frac{\\big\\|${S('patch', '\\x[\\mathcal N_p]')} - ${al}\\,${S('patch', '\\y_i[\\mathcal N_q]')}\\big\\|^2}{2${sg}^2}\\Big)`,
    after: ['Here \\(\\mathcal N_p\\) is the \\(P\\times P\\) square around pixel p. Kamb and Ganguli allow q to be any position, which is full translation equivariance. On images this small the full version is dominated by background patches, so this page limits the shift to a window of ±s pixels. The new samples are mosaics: each region is locally copied, but the combination is not a training image. The test in the readout only detects whole-image copies. Locality alone is one of the ways the network fails to fit the empirical field from step 2, and it is a failure that produces plausible new images.'],
    check: {
      q: 'Which setting turns this denoiser back into the empirical optimum of step 1?',
      opts: [['The patch is the whole image and s = 0', true], ['The smallest patch, P = 3', false], ['The largest shift window', false]],
      fb: { true: 'Right. When the patch covers the whole image and no shift is allowed, the weights of every pixel compare the full image \\(\\x_t\\) with each training image at its own position. All pixels then share the same softmax over the training images, which is the empirical denoiser. Select P = full and s = 0: every sample is a copy.', false: 'The empirical denoiser compares the whole noisy image with each whole training image, and uses the same weights for every pixel. This is the case where the patch is the full image and no shift is allowed.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>The exact optimum of the training loss copies, so generalization is a property of what a model fails to fit. The steps above showed three such failures that still give plausible samples: a limited resolution, a linear model and a convolutional architecture. §2.8 turns to the architectures used in practice. The consequence for post-training is the subject of §4 and §6: conditioning, guidance and long fine-tuning on small datasets sharpen the target and move a model back towards step 2.</p>
      <p>The earlier tutorials, <a href="../score-matching/">Denoising and score matching</a> and <a href="../flow-matching/">Flows: the continuity equation and flow matching</a>, derive the denoiser and the velocity used on this page.</p>`,
  },
];
