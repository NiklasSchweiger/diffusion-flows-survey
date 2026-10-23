// Content of the §2.10 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex,
// section "Autoregressive and hybrid visual generation"; S(id) tags a symbol from symbols.js, CHG marks
// the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), xi = S('xi'), xlt = S('xlt'), D = S('D'), mu = S('mu'), eta = S('eta'), pi = S('pi'), logp = S('logp');
const r = S('r'), b = S('b'), A = S('A'), K = S('K'), d = S('d'), z = S('z'), c = S('c'), eps = S('eps'), epsth = S('epsth');
const al = S('al'), sg = S('sg'), lam = S('lam'), n = S('n');

export const INTRO = `
  <div class="kicker">Survey §2.10 · Interactive tutorial</div>
  <h1>Autoregressive and hybrid visual generation</h1>
  <p>Diffusion generates all coordinates of an image together and refines them over noise levels. An autoregressive model generates them one after another, each conditioned exactly on the ones before. The survey places both on one axis: how the joint distribution is factorized into steps, and how each factor is parametrized. This tutorial follows that axis. It writes an image as a product of conditionals in any order, factorizes it over resolutions instead, measures what each sampler loses when it takes fewer steps, counts what quantizing an image into discrete tokens costs, gives one continuous token its own diffusion model, and draws the attention mask of a transformer that is trained with both losses.</p>
  <p>No step trains a network. Every canvas uses one exact image distribution: the eight 12 × 12 digits of the §2.7 tutorial as the means of a Gaussian mixture with pixel noise \\(\\eta = 0.3\\). Given the digit the pixels are independent, so every conditional, every coarse map and every noised marginal has a closed form, and all dependence between pixels runs through the posterior over the digit. Sample quality is measured with the exact log density: a sample is typical when \\(\\log p(\\x)\\) lies between the 1% and 99% quantiles of real draws. Random draws use fixed seeds, and the convention is that of the survey, with \\(t = 0\\) noise and \\(t = 1\\) data.</p>
  <ol class="toc">
    <li><a href="#chain">The chain rule in any order</a></li>
    <li><a href="#scale">Next-scale prediction</a></li>
    <li><a href="#axis">One axis: steps against dependence</a></li>
    <li><a href="#vq">Discrete tokens lose information</a></li>
    <li><a href="#mar">A diffusion head for one continuous token</a></li>
    <li><a href="#transfusion">One transformer, two losses</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'chain', kicker: 'Autoregressive factorization', title: 'The chain rule in any order', scene: 'chain',
    prose: [
      'Any joint density is the product of conditionals: the first pixel from its marginal, the second given the first, and so on. Nothing is approximated, and the order is free. An autoregressive model learns each factor, usually over discrete tokens with a cross-entropy loss, and generates by drawing the factors one after another. Its log-likelihood is the sum of the log conditionals, which is why autoregressive models have exact likelihoods.',
      'For the toy distribution each conditional is a mixture of eight Gaussians: pixel i is drawn from digit j with the posterior probability \\(\\pi_j\\) of that digit given the visible pixels, and the posterior is updated after every pixel. Move the slider or press Play to generate the image pixel by pixel. The bars show the posterior, the curve is the density of the next pixel, and the running sum of log conditionals ends exactly at the log density of the finished image. Switch to random order: the same identity holds for a different image, and scoring an image in another order gives the same total.',
    ],
    tex: `p(${x}) = \\prod_{i=1}^{${D}} p(${xi}\\mid ${xlt}), \\qquad p(${xi}\\mid ${xlt}) = \\sum_{j} ${pi}\\,\\N\\big(${xi};\\,(${mu})_i,\\,${eta}^2\\big)`,
    deriv: {
      have: ['The mixture \\(p(\\x) = \\frac18\\sum_j \\N(\\x;\\bm\\mu_j, \\eta^2 I)\\), whose components are products over the pixels.', 'The posterior \\(\\pi_j\\) of digit j given the visible pixels \\(\\x_{<i}\\).'],
      goal: 'p(x_i\\mid\\x_{<i}) = \\sum_j \\pi_j\\,\\N(x_i; (\\bm\\mu_j)_i, \\eta^2)',
      plan: 'Write the joint as a product with the chain rule, then compute one factor by conditioning on the digit and summing over it.',
      title: 'Why every conditional is a reweighted mixture', lines: [
        { tex: `p(\\x) = ${CHG('p(x_1)\\,p(x_2\\mid x_1)\\cdots p(x_D\\mid\\x_{<D})')}`, why: 'The chain rule of probability, applied in the chosen order. Any other order gives a different product with the same value.', uses: ['chain-rule'] },
        { tex: `p(x_i\\mid\\x_{<i}) = \\sum_j ${CHG('p(j\\mid\\x_{<i})')}\\,p(x_i\\mid j, \\x_{<i})`, why: 'Sum over the digit j, which is the hidden variable of the mixture.', uses: ['cond-exp'] },
        { tex: `p(x_i\\mid j, \\x_{<i}) = ${CHG('\\N(x_i; (\\bm\\mu_j)_i, \\eta^2)')}`, why: 'Given the digit the pixels are independent, so the visible pixels say nothing more about pixel i.', uses: ['gauss'] },
        { tex: `\\pi_j = p(j\\mid\\x_{<i}) \\propto ${CHG('\\prod_{i\' < i}\\N(x_{i\'}; (\\bm\\mu_j)_{i\'}, \\eta^2)')}`, why: 'Bayes\' rule with the uniform prior over the eight digits. After each pixel the posterior is multiplied by that pixel\'s likelihood under each digit and renormalized.', uses: ['bayes'] },
      ],
    },
    after: ['In raster order the first rows are background in every digit, so the posterior stays uniform for a while and becomes certain only after about 19 pixels. In random order four pixels are enough in this run. The order changes which pixels carry the decision, not the distribution that is sampled.'],
    check: {
      q: 'The raster run ends with a sum of log conditionals of −35.09. Its image is scored again in the random order of the other run. What total do you get?',
      answer: -35.093, tol: 0.01,
      fb: { true: 'Right. Every order is an exact factorization of the same density, so the total is \\(\\log p(\\x) = -35.09\\) whatever the order. Only the individual factors change.', false: 'The sum of the log conditionals in any order is the log of the product, which is \\(p(\\x)\\) by the chain rule.' },
    },
  },
  {
    id: 'scale', kicker: 'Factorizing over resolutions', title: 'Next-scale prediction', scene: 'scale',
    prose: [
      'VAR factorizes the image over resolutions instead of positions. A map \\(\\bm r_1\\) of one block average is generated first, then a 3 × 3 map given it, then 6 × 6, then the full 12 × 12 image. Each factor is a whole map, and all of its tokens are predicted in one step. Four sequential steps replace the 144 of the pixel-by-pixel model.',
      'In the toy each map is the block average of the image. Given the digit, the block averages are independent Gaussians with variance \\(\\eta^2/n\\) for blocks of n pixels, so the factor \\(p(\\bm r_s \\mid \\bm r_{<s})\\) has a closed form. The joint draw samples one digit from the posterior given the coarser maps and fills the whole map from it, which is exact. The per-token draw lets every token of the map pick its own digit from the same posterior, as a head that predicts the tokens of a step independently does. Compare the final samples and the share that passes the typicality test.',
    ],
    tex: `p(${x}) = \\prod_{s} p(${r}\\mid \\bm r_{<s}), \\qquad s \\in \\{1, 3, 6, 12\\}`,
    deriv: {
      have: ['Given the digit, the J children of one parent block are independent Gaussians \\(r_q = m_q + e_q\\) with \\(e_q \\sim \\N(0, v)\\).', 'The parent is their average.'],
      goal: 'r_q = m_q + (r_{\\mathrm{parent}} - \\bar m) + (e_q - \\bar e)',
      plan: 'Split the noise of each child into the part that the parent fixes and the part it does not.',
      title: 'Why the children of a block are drawn around the parent', lines: [
        { tex: `r_{\\mathrm{parent}} = \\frac1J\\sum_q r_q = ${CHG('\\bar m + \\bar e')}`, why: 'Average the children. The mean \\(\\bar m\\) is known for each digit, the noise \\(\\bar e\\) is what the parent reveals.', uses: ['expectation'] },
        { tex: `e_q = ${CHG('\\bar e + (e_q - \\bar e)')}, \\qquad \\mathrm{Cov}(\\bar e, e_q - \\bar e) = ${CHG('0')}`, why: 'For iid Gaussians the deviations from the mean are uncorrelated with the mean, and so independent of it.', uses: ['gauss'] },
        { tex: `r_q \\mid r_{\\mathrm{parent}} = m_q + ${CHG('(r_{\\mathrm{parent}} - \\bar m)')} + (e_q - \\bar e)`, why: 'Substitute \\(\\bar e = r_{\\mathrm{parent}} - \\bar m\\). Each child is shifted by the parent residual and gets the deviations, whose variance is \\(v(1 - 1/J)\\).', uses: ['cond-exp'] },
      ],
    },
    after: ['The joint draw gives 98.5% typical samples in four steps, the same share as real draws. The per-token draw gives 67%, and almost all of its failures fall below the band: blocks of one map take their strokes from different digits, which no real draw does. Next-scale prediction is exact only when the tokens of one scale are independent given the coarser maps, and here they are not, because they share the unknown digit.'],
    check: {
      q: 'How many tokens does the next-scale model generate in total over the four maps of sizes 1, 3, 6 and 12?',
      answer: 190, tol: 0.5,
      fb: { true: 'Right. \\(1 + 9 + 36 + 144 = 190\\) tokens, which is more than the 144 of the pixel-by-pixel model, but in 4 sequential steps instead of 144.', false: 'Add the sizes of the maps: \\(1^2 + 3^2 + 6^2 + 12^2\\).' },
    },
  },
  {
    id: 'axis', kicker: 'The cost of parallel steps', title: 'One axis: steps against dependence', scene: 'axis',
    prose: [
      'Autoregressive models, masked diffusion, next-scale prediction and continuous diffusion all generate in a number of sequential steps, and in each step they draw several coordinates at once. When a step draws its coordinates independently given what is visible, it samples the product of their conditionals instead of their joint conditional. The error of one step is the KL divergence between the two, the same quantity that limited parallel unmasking in §2.9.',
      'The plot shows the typical share of 300 samples against the number of sequential steps for four samplers: raster order with b pixels per step, random order with b pixels per step, next-scale prediction with one to four scales and per-token draws, and continuous diffusion run with N Euler steps of the exact probability-flow ODE. Pick a sampler and move the slider; the ring marks the current setting, and the images are eight of its samples. The bar under them splits the samples into too unlikely, typical and too likely.',
    ],
    tex: `\\mathrm{error\\ of\\ one\\ step} = \\KL\\Big(p(${A}\\mid \\x_V)\\ \\Big\\|\\ \\prod_{i \\in A} p(x_i\\mid \\x_V)\\Big)`,
    deriv: {
      have: ['The visible pixels \\(\\x_V\\) and the posterior \\(\\pi_j\\) given them.', 'A set A of pixels drawn in one step.'],
      goal: '\\KL = 0 \\text{ only when the posterior is a point mass (on pixels where the digits differ)}',
      plan: 'Write the joint conditional and the product of conditionals for the mixture and compare them.',
      title: 'When a parallel step is exact', lines: [
        { tex: `p(\\x_A\\mid\\x_V) = ${CHG('\\sum_j \\pi_j \\prod_{i\\in A}\\N(x_i; (\\bm\\mu_j)_i, \\eta^2)')}`, why: 'Condition on the digit as in step 1. All pixels of A come from the same digit.', uses: ['cond-exp'] },
        { tex: `\\prod_{i\\in A} p(x_i\\mid\\x_V) = ${CHG('\\prod_{i\\in A}\\sum_j \\pi_j\\,\\N(x_i; (\\bm\\mu_j)_i, \\eta^2)')}`, why: 'Each pixel is drawn from its own conditional, so each one picks its own digit.', uses: ['expectation'] },
        { tex: `\\pi = \\delta_{j^*} \\ \\Rightarrow\\ \\text{both} = ${CHG('\\prod_{i\\in A}\\N(x_i; (\\bm\\mu_{j^*})_i, \\eta^2)')}, \\quad \\KL = 0`, why: 'With a certain digit both sides are the same product. With two or more likely digits that differ on A, the product mixes strokes of different digits, which the joint never does, and the KL is positive.', uses: ['kl'] },
      ],
    },
    after: ['With four sequential steps the four samplers end far apart. Next-scale prediction keeps 63% of its samples typical, raster order 22%, random order 2%, and four Euler steps of the flow only 1%. The failures differ in kind. The autoregressive samplers fall below the band, because pixels drawn in one step take their strokes from different digits. The flow falls above it: few Euler steps follow the posterior mean, which averages the noise away and gives images that are too smooth to be real draws. Continuous diffusion becomes typical at about 32 steps, and raster order at about 36.'],
    check: {
      q: 'The visible pixels leave two digits with posterior 0.5 each, and they differ at two pixels that are drawn independently in one step. With what probability do both pixels come from the same digit?',
      answer: 0.5, tol: 0.005,
      fb: { true: 'Right. Each pixel picks a digit with probability 0.5, so they agree with probability \\(0.5^2 + 0.5^2 = 0.5\\). In the other half of the runs the two pixels come from different digits, which the joint conditional never produces.', false: 'They agree when both pick digit 1 or both pick digit 2: \\(\\sum_j \\pi_j^2\\).' },
    },
  },
  {
    id: 'vq', kicker: 'Quantization', title: 'Discrete tokens lose information', scene: 'vq',
    prose: [
      'An autoregressive model over discrete tokens needs a quantizer that maps each patch of the image to one of K codes. The model then predicts codes with a cross-entropy loss, and the decoder maps them back. Whatever the quantizer removes cannot be generated, however good the model of the codes is.',
      'Here the image is cut into 36 patches of 2 × 2 pixels, and each patch is replaced by the nearest of K codes found by k-means on the patches of 400 draws. The images show one test draw, its reconstruction from the codes and the error. The plot shows the error per pixel over 200 test draws against K on logarithmic axes, with the pixel noise \\(\\eta^2\\) marked.',
    ],
    tex: `\\E\\|\\x - Q_${K}(\\x)\\|^2 \\ \\propto\\ ${K}^{-2/${d}}`,
    deriv: {
      have: ['K codes that tile a region of volume V in d dimensions.', 'Each patch is replaced by the centre of its cell.'],
      goal: '\\text{error} \\propto K^{-2/d}',
      plan: 'Size of a cell, then the squared distance to its centre.',
      title: 'Why the error falls slowly in high dimension', lines: [
        { tex: `\\text{volume of a cell} \\approx ${CHG('V/K')}`, why: 'K cells of roughly equal size share the volume.', uses: [] },
        { tex: `\\text{side of a cell} \\approx ${CHG('(V/K)^{1/d}')}`, why: 'A cell of volume v in d dimensions has a diameter of order \\(v^{1/d}\\).', uses: [] },
        { tex: `\\text{squared error} \\approx ${CHG('(V/K)^{2/d} \\propto K^{-2/d}')}`, why: 'The distance from a point to the centre of its cell is of the order of the side, and the error is its square. For d = 4 doubling K reduces the error by only \\(2^{-1/2}\\), about 30%.', uses: [] },
      ],
    },
    after: ['The measured slope is −0.45 on the log axes, close to the −0.5 predicted for d = 4. The error falls below the pixel noise only at K = 32, and at K = 256 it is still 0.030 per pixel. Real images need far more information per token than this toy, which is why the quantizers of image models use large codebooks, residual or multi-scale codes, and why MAR, in the next step, keeps the tokens continuous.'],
    check: {
      q: 'How many bits does one image take when each of its 36 patches is stored as a code from a codebook of K = 256?',
      answer: 288, tol: 0.5,
      fb: { true: 'Right. Each code carries \\(\\log_2 256 = 8\\) bits, and 36 codes carry 288 bits. A continuous token has no such limit.', false: 'One code from 256 carries \\(\\log_2 256\\) bits. Multiply by the number of patches.' },
    },
  },
  {
    id: 'mar', kicker: 'Diffusion inside an autoregressive model', title: 'A diffusion head for one continuous token', scene: 'mar',
    prose: [
      'MAR keeps the tokens continuous and models each factor \\(p(\\bm z_k\\mid\\bm c_k)\\) with a small diffusion model. The transformer reads the tokens generated so far and outputs a context \\(\\bm c_k\\), and the head is trained with the denoising loss of §2.2 for one token, conditioned on that context. The gradients flow through the context into the transformer.',
      'The tokens are pairs of horizontally adjacent pixels, generated in a seeded random order. The image shows the first m tokens of a run; the ringed token is the one to predict, and you can click another. The plane shows its exact density given the m visible tokens, the paths that the probability-flow ODE of an ideal head takes from twelve noise points, and the cross where a regression head with a squared loss would put the token. Move m from 0 upward and watch the density go from eight modes to one.',
    ],
    tex: `\\mathcal L_{\\mathrm{MAR}} = \\E_{k,t,${eps}}\\big\\|${eps} - ${epsth}(${al}${z} + ${sg}${eps},\\,t,\\,${c})\\big\\|^2`,
    deriv: {
      have: ['The noised token \\(\\bm z_t = \\alpha_t\\bm z_k + \\sigma_t\\bm\\epsilon\\) and the context \\(\\bm c_k\\).', 'The regression lemma: the minimizer of a squared loss is the conditional mean.'],
      goal: '\\bm\\epsilon_\\theta^* = -\\sigma_t\\nabla\\log p_t(\\bm z_t\\mid\\bm c_k)',
      plan: 'Minimize the loss pointwise, then rewrite the conditional mean of the noise with Tweedie\'s formula.',
      title: 'What the head learns', lines: [
        { tex: `\\bm\\epsilon_\\theta^*(\\bm z_t, t, \\bm c_k) = ${CHG('\\E[\\bm\\epsilon\\mid\\bm z_t, \\bm c_k]')}`, why: 'The loss is a squared error, so its minimizer is the conditional mean of the target given the inputs.', uses: ['regression-lemma'] },
        { tex: `= ${CHG('\\frac{\\bm z_t - \\alpha_t\\E[\\bm z_k\\mid\\bm z_t, \\bm c_k]}{\\sigma_t}')}`, why: 'Solve \\(\\bm z_t = \\alpha_t\\bm z_k + \\sigma_t\\bm\\epsilon\\) for the noise and take the conditional mean.', uses: ['cond-exp'] },
        { tex: `= ${CHG('-\\sigma_t\\nabla\\log p_t(\\bm z_t\\mid\\bm c_k)')}`, why: 'Tweedie\'s formula for the noised conditional distribution. The head is the score of the token given its context, and sampling with it draws from \\(p(\\bm z_k\\mid\\bm c_k)\\).', uses: ['tweedie'] },
      ],
    },
    after: ['With no visible tokens the ringed token has eight possible values, one per digit, and the regression head predicts their average, a value that no digit has. After about four tokens the posterior is nearly certain and the two heads almost agree, but the regression head still drops the pixel noise. The images under the plane show the consequence. The diffusion head generates noisy draws of several digits, three of the four inside the typical band. The regression head generates noise-free images with a log density of about 36, far above the typical band that ends at −14.8, because the noise that every real draw carries is never sampled.'],
    check: {
      q: 'Before any token is visible, a pixel is ink (+1) in four digits and background (−1) in the other four, all equally likely. What value does a regression head predict for it?',
      answer: 0, tol: 0.01,
      fb: { true: 'Right. The conditional mean is \\(\\frac12(+1) + \\frac12(-1) = 0\\), a grey value between the two modes that neither digit produces. A diffusion head samples +1 or −1 instead.', false: 'A squared loss is minimized by the conditional mean, here the average of +1 and −1 with equal weights.' },
    },
  },
  {
    id: 'transfusion', kicker: 'Unified models', title: 'One transformer, two losses', scene: 'transfusion',
    prose: [
      'Transfusion puts a full diffusion model inside a language model. One transformer reads a sequence in which text tokens and image patches are interleaved. Text tokens are trained with next-token prediction and image patches with a diffusion loss, and the two losses are added with a weight \\(\\lambda\\). What makes this work is the attention mask: attention is causal across the sequence, so text can be generated after an image and an image after text, and bidirectional within each image, so the patches of one image are denoised together as in an ordinary diffusion model.',
      'The matrix is the attention mask of a sequence of nine text tokens and two images of nine patches each. Row i shows which tokens position i may attend to. Switch between a causal mask, the Transfusion mask and a fully bidirectional one, and hover a row to see what that token reads.',
    ],
    tex: `\\mathcal L = \\mathcal L_{\\mathrm{LM}} + ${lam}\\,\\mathcal L_{\\mathrm{diff}}, \\qquad \\mathcal L_{\\mathrm{LM}} = -\\sum_{\\text{text}}\\log p_\\theta(y_k\\mid y_{<k}), \\quad \\mathcal L_{\\mathrm{diff}} = \\E\\big\\|${eps} - ${epsth}(${al}\\x + ${sg}${eps}, t, \\text{prefix})\\big\\|^2`,
    after: ['The bidirectional block of each image is what lets the patches of one image be drawn jointly, which steps 2 and 3 showed to matter. A fully bidirectional mask would let text attend to the images that follow it, so text could no longer be generated left to right. The unified models of the survey differ mainly in where they place this boundary: Show-o2 and Transfusion run a diffusion or flow loss on the image positions of one transformer, Emu3.5 and UniAR keep discrete image tokens and next-token prediction, and UniDDT hands the image to a separate diffusion transformer that reads the language model.'],
    check: {
      q: 'Compared with a causal mask, how many query–key pairs does bidirectional attention add within one image of n = 9 patches?',
      answer: 36, tol: 0.5,
      fb: { true: 'Right. The causal mask already allows the n(n + 1)/2 pairs on and below the diagonal of the block; bidirectional attention adds the \\(n(n - 1)/2 = 36\\) pairs above it.', false: 'A block of n × n pairs has n(n + 1)/2 on and below the diagonal. The rest, \\(n(n - 1)/2\\), are added.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>Autoregressive models, masked diffusion, next-scale prediction and continuous diffusion are different factorizations of the same joint distribution, and they differ in how many sequential steps they take and in how much dependence each step must capture in parallel. When a step draws dependent coordinates independently, samples mix modes; when a continuous sampler takes too few steps, samples are smoothed toward the mean. A diffusion head removes the quantization of discrete tokens, and a mask with causal and bidirectional parts lets one transformer generate both text and images. <a href="../guidance/">§2.11</a> turns to conditioning and guidance, which shape the conditional \\(p(\\x\\mid\\bm c)\\) that every factor on this page depends on.</p>
      <p>The earlier tutorials <a href="../discrete/">Discrete and language diffusion</a>, <a href="../score-matching/">Denoising and score matching</a> and <a href="../architectures/">Representations and architectures</a> develop the pieces used on this page.</p>`,
  },
];
