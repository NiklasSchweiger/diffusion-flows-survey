// Content of the §2.0 primer: one entry per step. It comes before §2.1 and introduces the objects the
// later tutorials take for granted: data points as vectors, a dataset as samples, densities, Gaussian
// fits and mixtures. S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';
import { POP, H_DATA, KL_GAUSS } from './sim.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const xi = S('xi'), sum = `\\frac{1}{${S('n')}}\\sum_{i=1}^{${S('n')}}`;
const f2 = (v) => v.toFixed(2);

export const INTRO = `
  <div class="kicker">Survey §2.0 · Interactive primer</div>
  <h1>Data and distributions</h1>
  <p>The tutorials that follow take three things for granted: that a data point is a vector, that a dataset is a set of samples from an unknown distribution, and that a model is a density we can fit to those samples. This primer builds those three ideas from the ground up, one formula at a time. If you already know what a maximum-likelihood Gaussian fit is, you can go straight to <a href="../ebm-langevin/">§2.1</a>.</p>
  <p>Every symbol can be hovered or tapped to see what it is, what shape it has and where it appears in the plot. The button "Symbols" in the top bar opens the full list. From step 2 on, the data come from the same one-dimensional mixture with three modes that the later tutorials use, so its density is known exactly and every fit can be compared with the truth.</p>
  <ol class="toc">
    <li><a href="#vector">A data point is a vector</a></li>
    <li><a href="#samples">A dataset is a sample from a distribution</a></li>
    <li><a href="#density">What a density means</a></li>
    <li><a href="#gauss1d">Fitting a Gaussian</a></li>
    <li><a href="#gauss2d">Gaussians in more than one dimension</a></li>
    <li><a href="#onegauss">One Gaussian is not enough</a></li>
    <li><a href="#mixture">A mixture of Gaussians</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'vector', kicker: 'Data as vectors', title: 'A data point is a vector', scene: 'objects',
    prose: [
      'A generative model never sees a picture or a molecule. It sees a list of numbers, and a convention says which number describes which part of the object. Choose an object with the buttons and move the pointer over it or over the vector on the right. The highlighted entries are the numbers that describe the part under the pointer.',
      'The image stores one brightness per pixel, row after row. The molecule stores, for each heavy atom, three coordinates in ångström and a one-hot code of the element. The protein stores, for each residue, the position of its Cα atom and the rotation of its backbone frame. The sentence stores one token id per word. The molecule and the protein are real: caffeine from PubChem and the first sixteen residues of ubiquitin from the Protein Data Bank.',
    ],
    tex: `${S('x')} = (${S('xj', 'x_1')}, ${S('xj', 'x_2')}, \\dots, ${S('xj', `x_{${S('d')}}`)}) \\in ${'\\R'}^{${S('d')}}`,
    after: ['The choice of vector matters. The entries of the image are bounded and on a grid, the coordinates of the molecule change when the molecule is rotated, and the token ids are labels with no order at all, so adding noise to them makes no sense without first mapping them to vectors. Later chapters of the survey treat each case. The rest of this primer uses d = 1 and d = 2, where a distribution can be drawn.'],
    check: {
      q: 'A colour photograph of 256 × 256 pixels stores three numbers per pixel (red, green and blue). What is d?',
      answer: 196608, tol: 0.5,
      fb: {
        true: 'Right. 256 × 256 × 3 = 196 608. A generative model for such images works with densities on a space of that dimension.',
        false: 'Count the numbers: 256 rows, 256 columns and three colour channels per pixel. Multiply them.',
      },
    },
  },
  {
    id: 'samples', kicker: 'Datasets', title: 'A dataset is a sample from a distribution', scene: 'samples',
    prose: [
      'We assume that the data points were drawn independently from one distribution \\(\\pdata\\). Nobody gives us \\(\\pdata\\); we only have the n samples. The distribution that puts mass 1/n on each sample is the empirical distribution \\(\\hat p_n\\). It is all the information the dataset contains.',
      'The dots are the samples and the dashed curve is the true \\(\\pdata\\). The bars form a histogram of the samples, scaled so that their total area is one. Increase n and the histogram approaches the curve. Change the bin width h: narrow bins follow the samples closely but are noisy, wide bins are smooth but blur the modes.',
    ],
    tex: `${xi} \\sim ${S('pdata')} \\ \\text{for } i = 1, \\dots, ${S('n')}, \\qquad ${S('phat')}(${S('x')}) = ${sum} ${S('delta')}(${S('x')} - ${xi})`,
    after: ['A generative model has to do better than \\(\\hat p_n\\), which can only reproduce the training points. It has to put mass between and around the samples in the places where \\(\\pdata\\) has mass. The histogram is the simplest such model, and the bin width already shows the central trade-off: how much to trust each sample against how smooth the result should be.'],
    check: {
      q: 'Set n = 30. With h = 0.15 the histogram has many separate spikes, with h = 1 it shows one or two broad bumps. Which statement is right?',
      opts: [
        ['The data have changed between the two settings', false],
        ['Both histograms are estimates of the same density; with 30 samples, neither bin width can show all three modes reliably', true],
        ['The narrow bins are correct because they are closer to the samples', false],
      ],
      fb: {
        true: 'Right. The samples are the same in both cases. With few samples, narrow bins mostly show where individual points happen to lie, and wide bins merge neighbouring modes. Only more data removes the trade-off.',
        false: 'The samples do not change when h changes; only the way they are counted does. Compare both histograms with the dashed curve and ask which features of it each one shows.',
      },
    },
  },
  {
    id: 'density', kicker: 'Densities', title: 'What a density means', scene: 'interval',
    prose: [
      'A density is not a probability. The probability of a single point is zero for a continuous variable. What a density gives is the probability of a region, as the area under the curve over that region.',
      'Move the sliders a and b. The shaded area is the exact probability that a sample lands in (a, b), and the readout compares it with the fraction of 1000 samples that did. The right panel shows a single Gaussian whose width you can change. Make it narrow and its peak rises above 1, while its area stays exactly 1.',
    ],
    tex: `${S('P')}(${S('a')} < ${S('x')} < ${S('b')}) = \\int_{${S('a')}}^{${S('b')}} ${S('pdata')}(${S('x')})\\,\\dd ${S('x')} \\;\\approx\\; \\frac{\\#\\{i : ${S('a')} < ${xi} < ${S('b')}\\}}{${S('n')}}`,
    after: ['The units make the point. If x is measured in ångström, a density on x is measured in probability per ångström. Rescaling x by a factor of 10 divides the density by 10 and leaves every probability unchanged. Log-densities, which every later tutorial uses, therefore depend on the units, but differences of log-densities between two models of the same data do not.'],
    check: {
      q: 'Can a density be larger than 1?',
      opts: [
        ['No, because probabilities are at most 1', false],
        ['Yes, as long as its total area is 1', true],
        ['Only in more than one dimension', false],
      ],
      fb: {
        true: 'Right. A Gaussian with σ = 0.1 has peak height 1/(√(2π) · 0.1) ≈ 3.99. Only areas are probabilities, and the area under the whole curve is still 1.',
        false: 'Set σ = 0.1 in the right panel and read the peak height. The density is a probability per unit length, so it can be large over a short interval as long as the total area stays 1.',
      },
    },
  },
  {
    id: 'gauss1d', kicker: 'Maximum likelihood', title: 'Fitting a Gaussian', scene: 'fit1d',
    prose: [
      'The simplest model with a density is a Gaussian with two parameters, the mean μ and the standard deviation σ. To judge how well a choice of parameters explains the data, we average the log-density the model gives to each sample. This average log-likelihood \\(\\ell\\) is large when the model puts high density where the samples are.',
      'Move μ and σ, or click a point on the map on the right, which shows \\(\\ell\\) for every pair. A σ that is too small makes the samples in the tails very unlikely; a σ that is too large spreads the density thin everywhere. The cross marks the maximum, and "Fit" jumps there. Step through the derivation below to see why the maximum sits at the sample mean and the sample variance.',
    ],
    tex: `${S('Nx')}(${S('x')}; ${S('mu')}, ${S('sigma')}^2) = \\frac{1}{\\sqrt{2\\pi}\\,${S('sigma')}}\\,e^{-\\frac{(${S('x')} - ${S('mu')})^2}{2${S('sigma')}^2}}, \\qquad ${S('ell')}(${S('mu')}, ${S('sigma')}) = ${sum} \\log ${S('Nx')}(${xi}; ${S('mu')}, ${S('sigma')}^2)`,
    deriv: {
      have: ['The Gaussian density \\(\\N(x; \\mu, \\sigma^2)\\) and the average log-likelihood \\(\\ell(\\mu, \\sigma)\\) above.', 'n samples \\(x^{(1)}, \\dots, x^{(n)}\\).'],
      goal: `\\hat{${S('mu')}} = ${sum} ${xi}, \\qquad \\hat{${S('sigma')}}^2 = ${sum} (${xi} - \\hat{${S('mu')}})^2`,
      plan: 'Take the logarithm of the Gaussian so that the exponential disappears. Then set the derivative of \\(\\ell\\) with respect to each parameter to zero and solve.',
      lines: [
        { tex: `${S('ell')}(${S('mu')}, ${S('sigma')}) = ${sum} \\log ${S('Nx')}(${xi}; ${S('mu')}, ${S('sigma')}^2)`, why: 'Definition of the average log-likelihood.' },
        { tex: `= ${CHG(`-\\tfrac12\\log(2\\pi) - \\log ${S('sigma')} - \\frac{1}{2${S('sigma')}^2}${sum}(${xi} - ${S('mu')})^2`)}`, why: 'The log of a product is the sum of the logs, and \\(\\log e^{u} = u\\). The first two terms do not depend on i, so their average is themselves.', uses: ['log-rules'] },
        { tex: `\\frac{\\partial ${S('ell')}}{\\partial ${S('mu')}} = ${CHG(`\\frac{1}{${S('sigma')}^2}${sum}(${xi} - ${S('mu')})`)} = 0 \\;\\Rightarrow\\; \\hat{${S('mu')}} = ${sum} ${xi}`, why: 'Only the last term depends on μ. The derivative of \\((x - \\mu)^2\\) with respect to μ is \\(-2(x - \\mu)\\). Setting the sum to zero gives n μ = Σ x⁽ⁱ⁾.', uses: ['chain-rule'] },
        { tex: `\\frac{\\partial ${S('ell')}}{\\partial ${S('sigma')}} = ${CHG(`-\\frac{1}{${S('sigma')}} + \\frac{1}{${S('sigma')}^3}${sum}(${xi} - \\hat{${S('mu')}})^2`)} = 0 \\;\\Rightarrow\\; \\hat{${S('sigma')}}^2 = ${sum}(${xi} - \\hat{${S('mu')}})^2`, why: 'Differentiate in σ at μ = μ̂: \\(\\log\\sigma\\) gives \\(1/\\sigma\\) and \\(1/(2\\sigma^2)\\) gives \\(-1/\\sigma^3\\). Multiply by \\(\\sigma^3\\) and solve.', uses: ['chain-rule'] },
      ],
    },
    after: ['Maximum likelihood is the principle behind most generative models. The models of later chapters have millions of parameters instead of two and no closed-form maximum, so they climb \\(\\ell\\) by gradient ascent, but the quantity they climb is the same.'],
    check: {
      q: 'Three samples are 1, 2 and 3. What is the maximum-likelihood variance σ̂²?',
      answer: 2 / 3, tol: 0.01,
      fb: {
        true: 'Right. μ̂ = 2 and σ̂² = ((1 − 2)² + 0 + (3 − 2)²)/3 = 2/3 ≈ 0.667. Maximum likelihood divides by n, not by n − 1 as the unbiased estimator does.',
        false: 'First compute μ̂ = (1 + 2 + 3)/3. Then average the squared distances to μ̂, dividing by n = 3.',
      },
    },
  },
  {
    id: 'gauss2d', kicker: 'More dimensions', title: 'Gaussians in more than one dimension', scene: 'gauss2d',
    prose: [
      'In d dimensions the mean is a vector and the width becomes a covariance matrix Σ. Its diagonal holds the variance of each coordinate, and its off-diagonal entries say how two coordinates vary together. The level sets of the density are ellipses (ellipsoids for d > 2), whose axes are the eigenvectors of Σ.',
      'The dots are 600 samples of the two-dimensional mixture used in §2.1 and §2.2. The ellipses are the 1σ and 2σ contours of a Gaussian centred at the sample mean. Change the two standard deviations and the correlation ρ, and watch the average log-likelihood. "Fit" sets them to the sample values.',
    ],
    tex: `${S('Nx')}(${S('x')}; \\bm{${S('mu')}}, ${S('Sigma')}) = \\frac{1}{\\sqrt{(2\\pi)^{${S('d')}}\\det ${S('Sigma')}}}\\,\\exp\\!\\Big(-\\tfrac12(${S('x')} - \\bm{${S('mu')}})^\\top ${S('Sigma')}^{-1}(${S('x')} - \\bm{${S('mu')}})\\Big)`,
    after: ['The maximum-likelihood fit is again the sample mean and the sample covariance, by the same derivation as in step 4 with vectors. The cost lies in the number of parameters: d for the mean and d(d + 1)/2 for the symmetric matrix Σ. Here that is 2 + 3 = 5. Step 7 asks what it is for an image.'],
    check: {
      q: 'Set ρ = 0 and σ₁ = σ₂. What shape is the 1σ contour?',
      opts: [['An ellipse tilted by 45°', false], ['A circle', true], ['A line', false]],
      fb: {
        true: 'Right. With Σ = σ²I the quadratic form is ‖x − μ‖²/σ², which is constant on circles. Such a Gaussian is called isotropic, and the noise of every diffusion model on this site is of this kind.',
        false: 'With ρ = 0 and equal variances, Σ = σ²I, and the exponent only depends on the distance ‖x − μ‖. Set the sliders and look.',
      },
    },
  },
  {
    id: 'onegauss', kicker: 'Model mismatch', title: 'One Gaussian is not enough', scene: 'onegauss',
    prose: [
      `The data of step 2 have three modes. Fit a single Gaussian to infinitely many samples and you get the mean and variance of \\(\\pdata\\), here \\(\\hat\\mu = ${f2(POP.m)}\\) and \\(\\hat\\sigma = ${f2(POP.s)}\\). That Gaussian is the bold curve. It covers all three modes, and to do so it puts a lot of mass in the gaps between them, shaded in the upper panel, where the data have little.`,
      'The KL divergence measures the mismatch in one number. The lower panel shows its integrand, \\(\\pdata\\log(\\pdata/q)\\), whose area is the KL divergence. Move the pointer to compare the two densities at one point. Step through the derivation below to see why maximizing the likelihood is the same as minimizing this divergence.',
    ],
    tex: `${S('KL')}(${S('pdata')}\\,\\|\\,${S('q')}) = \\int ${S('pdata')}(${S('x')})\\log\\frac{${S('pdata')}(${S('x')})}{${S('q')}(${S('x')})}\\,\\dd ${S('x')}`,
    deriv: {
      have: ['The KL divergence above, which is zero only when \\(q = \\pdata\\).', 'The average log-likelihood \\(\\ell\\) of step 4, for a model density q.'],
      goal: `\\arg\\max_{${S('q')}}\\;\\E_{${S('pdata')}}[\\log ${S('q')}(${S('x')})] = \\arg\\min_{${S('q')}}\\;${S('KL')}(${S('pdata')}\\,\\|\\,${S('q')})`,
      plan: 'Split the logarithm of the ratio. One part does not depend on the model; the other is the expected log-likelihood, which \\(\\ell\\) estimates.',
      lines: [
        { tex: `${S('KL')}(${S('pdata')}\\,\\|\\,${S('q')}) = \\int ${S('pdata')}\\log\\frac{${S('pdata')}}{${S('q')}}\\,\\dd ${S('x')}`, why: 'Definition.', uses: ['kl'] },
        { tex: `= ${CHG(`\\int ${S('pdata')}\\log ${S('pdata')}\\,\\dd ${S('x')} - \\int ${S('pdata')}\\log ${S('q')}\\,\\dd ${S('x')}`)}`, why: 'The log of a quotient is a difference.', uses: ['log-rules'] },
        { tex: `= ${CHG(`-${S('H')}(${S('pdata')})`)} - ${CHG(`\\E_{${S('pdata')}}[\\log ${S('q')}(${S('x')})]`)}`, why: 'The first integral is minus the entropy H of the data, which does not depend on q. The second is an expectation under the data.', uses: ['expectation'] },
        { tex: `${CHG(`${S('ell')} = ${sum}\\log ${S('q')}(${xi})`)} \\;\\to\\; \\E_{${S('pdata')}}[\\log ${S('q')}(${S('x')})] \\quad (${S('n')}\\to\\infty)`, why: 'The average over samples estimates the expectation, by the law of large numbers. So maximizing ℓ over q minimizes the KL divergence, up to a constant and sampling error.', uses: ['expectation'] },
      ],
    },
    after: [`For this data, \\(H(\\pdata) = ${f2(H_DATA)}\\) nats and the best single Gaussian reaches \\(\\KL = ${f2(KL_GAUSS)}\\) nats. That is the smallest KL any Gaussian can reach, and the remainder is the price of the wrong model family, not of too few samples.`],
    check: {
      q: `At \\(x = \\hat\\mu\\), between the modes, the fitted Gaussian has density 0.27 and the data only 0.15. Why does maximum likelihood accept this?`,
      opts: [
        ['The fit has not converged', false],
        ['KL(p_data ‖ q) weighs errors by p_data, so overestimating q where the data are rare costs little, while underestimating q at a mode would cost a lot', true],
        ['The two densities are equal on average', false],
      ],
      fb: {
        true: 'Right. Every term of the integrand is weighted by p_data. A Gaussian that left out a mode would make log q very negative where the data are, and that is punished heavily. Maximum likelihood is therefore mass-covering: it would rather spread mass into the gaps than miss a mode.',
        false: 'The fit is exact. Look at the lower panel: the integrand is small in the gaps because p_data is small there, and it would be large near a mode if q left that mode out. Which kind of error does this weighting forgive?',
      },
    },
  },
  {
    id: 'mixture', kicker: 'Mixtures', title: 'A mixture of Gaussians', scene: 'mixture',
    prose: [
      'A mixture of K Gaussians is a weighted sum of K bumps. Each component has its own mean, width and weight, and the weights sum to one. With enough components a mixture can approximate any smooth density on the line, and the data of this page are in fact a mixture of three.',
      'The expectation-maximization algorithm fits a mixture to samples. It alternates two updates: it computes for each sample how much each component explains it, then refits each component to the samples it explains, weighted by those shares. Each round cannot decrease the likelihood. Choose K and press Play to watch EM fit 200 samples from a seeded start. The lower curve is the KL divergence to \\(\\pdata\\) after each iteration.',
    ],
    tex: `${S('q')}(${S('x')}) = \\sum_{k=1}^{${S('K')}} ${S('pik')}\\,${S('Nx')}(${S('x')}; ${S('muk')}, ${S('sk')}^2), \\qquad \\sum_{k=1}^{${S('K')}} ${S('pik')} = 1`,
    after: ['At K = 3 the fit lands close to the true parameters. At K = 4 and 5 it can split one mode or chase a cluster of samples, and its KL to \\(\\pdata\\) gets slightly worse even though its likelihood on the 200 samples improves. That is overfitting, the same trade-off as the bin width of step 2.'],
    check: {
      q: 'One Gaussian with a full covariance on the 16 × 16 images of step 1: how many parameters does it have?',
      answer: 33152, tol: 0.5,
      fb: {
        true: 'Right. d + d(d + 1)/2 = 256 + 256 · 257/2 = 33 152 for a single component on a tiny image. For a 256 × 256 colour image the covariance alone has about 1.9 · 10¹⁰ entries. Mixtures of full Gaussians do not scale, which is why the models of the survey represent a density through a neural network instead.',
        false: 'Use d + d(d + 1)/2 from step 5 with d = 256.',
      },
    },
    closing: `<h3>Where this leads</h3>
      <p>Everything later in the survey builds on these objects. §2.1 starts from a density written as an energy and asks how to sample from it when it cannot be normalized. §2.2 learns the score of \\(\\pdata\\) from samples, and §2.3 adds Gaussian noise of growing size to the data until only a standard Gaussian is left, which is the noise end \\(t = 0\\) of every model on this site.</p>
      <p><a href="../ebm-langevin/">Continue with §2.1, Energies, scores and Langevin dynamics</a>.</p>`,
  },
];
