// Content of the §2.8 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex,
// section "Representations and architectures"; S(id) tags a symbol from symbols.js, CHG marks the part
// of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), t = S('t'), al = S('alpha'), sg = S('sig'), N = S('N'), f = S('f'), p = S('p'), b = S('b'), ts = S('ts');
const hx = S('hx1'), feat = S('feat'), Pk = S('Pk'), r = S('r'), D = S('D'), x0 = S('x0'), x1 = S('x1'), u = S('vel'), lam = S('lam');
const q = S('q'), key = S('key'), val = S('val'), tau = S('tau'), yi = S('y');

export const INTRO = `
  <div class="kicker">Survey §2.8 · Interactive tutorial</div>
  <h1>Representations and architectures</h1>
  <p>The mathematics of the earlier tutorials holds in any Euclidean space. In practice two engineering choices decide what a model can learn at a given cost: the space in which the diffusion runs and the network that parametrizes the field. This tutorial follows the survey through both. It counts what an autoencoder and patching save, shows why the noise schedule has to change with the resolution, builds an encoder that reconstructs perfectly and still makes generation worse, reads representation alignment as a Tweedie estimate, asks which prediction target a narrow network can learn, writes the empirical denoiser of §2.7 as an attention layer, and compares the ways text and image tokens meet in attention.</p>
  <p>No step trains a network. Each plot is exact arithmetic, an exact Gaussian or Gaussian-mixture computation, or the best linear predictor of a given width, which depends only on the means and covariances of the data and is computed in closed form. Where a stand-in replaces a trained model, the page says so. The convention is that of the survey: \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), where \\(t = 0\\) is noise and \\(t = 1\\) is data, on the linear path \\(\\alpha_t = t\\), \\(\\sigma_t = 1 - t\\).</p>
  <ol class="toc">
    <li><a href="#tokens">From pixels to tokens</a></li>
    <li><a href="#shift">The same noise level hides less in more pixels</a></li>
    <li><a href="#latent">Perfect reconstruction, different generation</a></li>
    <li><a href="#repa">Representation alignment through Tweedie</a></li>
    <li><a href="#pixels">Back to pixels: which target a narrow network can learn</a></li>
    <li><a href="#attention">Attention is the empirical denoiser</a></li>
    <li><a href="#mmdit">Joint attention in MMDiT</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'tokens', kicker: 'The cost of a training step', title: 'From pixels to tokens', scene: 'tokens',
    prose: [
      'A 1024 × 1024 RGB image has about three million coordinates. Latent diffusion first trains an autoencoder and runs the diffusion on its output. The autoencoder of Stable Diffusion maps each 8 × 8 block of pixels to one position with four channels, which reduces the dimension by a factor of 48. A diffusion transformer then groups \\(p\\times p\\) latent positions into one token. For an image or video of height H, width W and T frames, the number of tokens is the formula below.',
      'Attention compares every token with every other token, so the cost of each layer grows like \\(N^2\\). The plot draws the latent grid and the tokens over a small picture, and the bars on the right compare the number of coordinates, the number of tokens and the number of attention pairs on a logarithmic scale. Try the presets. A deep compression autoencoder with \\(f = 32\\) and patches of size 1 gives a quarter of the tokens of the original autoencoder, and a video autoencoder that also merges eight frames keeps a 121-frame video at a manageable length.',
    ],
    tex: `${N} = \\frac{${S('HW', 'H')}\\,${S('HW', 'W')}\\,(${S('T')}/${S('fT')})}{${f}^2\\,${p}^2}`,
    after: ['The compression factor sets the price of every training step, which is why deep compression autoencoders matter at high resolution. It also changes what the diffusion model sees: the latent space has its own geometry, and step 3 shows that this geometry matters as much as the reconstruction error.'],
    check: {
      q: 'You double the downsampling factor f and keep everything else fixed. By what factor does the number of attention pairs \\(N^2\\) fall?',
      answer: 16, tol: 0,
      fb: { true: 'Right. N falls by \\(2^2 = 4\\), because f enters squared, and the number of pairs falls by \\(4^2 = 16\\). Move f from 8 to 16 at a fixed resolution and compare the last bar.', false: 'Doubling f divides N by \\(2^2 = 4\\), since f enters the formula squared. The number of pairs is \\(N^2\\), so it falls by \\(4^2 = 16\\).' },
    },
  },
  {
    id: 'shift', kicker: 'Schedules depend on the dimension', title: 'The same noise level hides less in more pixels', scene: 'shift',
    prose: [
      'Take a small image and upsample it by a factor \\(b\\), so that every pixel becomes a \\(b\\times b\\) block. At the same time \\(t\\), add independent noise to every pixel. Averaging \\(\\x_t\\) over each block gives back the small image plus noise, but the average of \\(b^2\\) independent noise values has standard deviation \\(\\sigma_t/b\\). The coarse shape of the image is therefore visible much earlier at high resolution, and the steps of a schedule designed for small images are wasted at the start.',
      'The left image is \\(\\x_t\\) at the base resolution of 12 × 12 pixels, the middle one is \\(\\x_t\\) of the upsampled digit, and the right one is the block average of the middle image. Move \\(b\\) and \\(t\\). Then switch on "shift the schedule". The high-resolution image is then noised at the shifted time \\(t\'\\) of the formula, and the block average looks like the base image again.',
    ],
    tex: `\\mathrm{SNR}_{${b}}(${t}) = \\frac{${b}^2\\,${al}^2}{${sg}^2}, \\qquad ${ts} = \\frac{${t}}{${t} + ${b}\\,(1 - ${t})}`,
    deriv: {
      have: ['\\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) with independent \\(\\N(0, 1)\\) noise in every pixel.', 'Every pixel of a \\(b\\times b\\) block of the upsampled \\(\\x_1\\) has the value of one base pixel.'],
      goal: `${ts} = \\frac{t}{t + b(1 - t)}`,
      plan: 'Compute the variance of a block average, then solve for the time at which the coarse content has the signal-to-noise ratio of the base image.',
      title: 'Why the schedule shifts with the resolution', lines: [
        { tex: `\\frac{1}{${b}^2}\\sum_{\\text{block}} ${x} = \\alpha_t\\,(\\text{base pixel}) + \\sigma_t\\,${CHG('\\frac{1}{b^2}\\sum_{\\text{block}}\\x_0')}`, why: 'Average both terms of the interpolation over the block. The signal is the same in every pixel of the block.', uses: ['expectation'] },
        { tex: `\\operatorname{Var}\\Big[\\frac{1}{b^2}\\sum_{\\text{block}}\\x_0\\Big] = \\frac{b^2}{b^4} = \\frac{1}{b^2} \\quad\\Rightarrow\\quad \\mathrm{SNR}_b(t) = ${CHG('\\frac{b^2\\alpha_t^2}{\\sigma_t^2}')}`, why: 'The variance of a sum of independent terms is the sum of their variances, here \\(b^2\\), and dividing by \\(b^2\\) divides the variance by \\(b^4\\).', uses: ['expand-square'] },
        { tex: `\\frac{b^2 t'^2}{(1 - t')^2} = \\frac{t^2}{(1 - t)^2} \\quad\\Leftrightarrow\\quad \\frac{t'}{1 - t'} = \\frac{t}{b\\,(1 - t)} \\quad\\Leftrightarrow\\quad t' = ${CHG('\\frac{t}{t + b(1 - t)}')}`, why: 'Set the SNR of the high-resolution image at \\(t\'\\) equal to the SNR of the base image at \\(t\\), take the square root and solve for \\(t\'\\).', uses: ['substitution'] },
      ],
    },
    after: ['This is the time shift of Stable Diffusion 3, which chooses the factor according to the resolution. The same argument holds for any increase in the number of coordinates that carry the same content, which is why representation autoencoders with high-dimensional latents also need a schedule shifted with the dimension. The page uses independent noise in each pixel, which is the forward process of every model in this chapter.'],
    check: {
      q: 'At \\(b = 4\\) and \\(t = 0.5\\), what is the standard deviation of the noise in the block average?',
      answer: 0.125, tol: 0.006,
      fb: { true: 'Right. \\(\\sigma_t = 0.5\\), and averaging 16 independent values divides the standard deviation by 4, which gives 0.125. The readout shows the value measured on the fixed noise of the page.', false: 'At \\(t = 0.5\\) the noise coefficient is \\(\\sigma_t = 0.5\\). A block of side 4 has 16 pixels, and the average of 16 independent values has standard deviation \\(\\sigma_t/\\sqrt{16} = 0.5/4 = 0.125\\).' },
    },
  },
  {
    id: 'latent', kicker: 'What makes a latent space easy to model', title: 'Perfect reconstruction, different generation', scene: 'latent',
    prose: [
      'Better reconstruction does not imply better generation. The survey cites tokenizer studies in which generation quality follows the geometry of the latent space, in particular its local continuity, more closely than it follows reconstruction. This step isolates that effect with two encoders that both reconstruct perfectly. The first is the identity. The second cuts the horizontal axis into \\(k\\) strips and moves each strip to another position. On each strip it is a translation, so the decoder that moves the strips back recovers every point exactly.',
      'Both latent spaces get the same generative model, the smoothed empirical distribution of §2.7 with bandwidth \\(h\\), fitted to the latents of the same 400 training points. It stands in for a trained model that cannot resolve details smaller than h. The left panel shows its decoded samples for the identity encoder. The middle panel shows the latents of the strip encoder and the samples of the model in latent space, and the right panel shows those samples after decoding. Increase \\(k\\) and \\(h\\). Every sample that the model places across a strip boundary is decoded into the wrong place, far from the data.',
    ],
    tex: `${S('Dec')}(${S('Enc')}(\\x)) = \\x \\quad\\text{for both encoders}, \\qquad \\text{sample} = ${S('Dec')}(${S('z')}),\\quad ${S('z')} \\sim \\frac1n\\sum_i \\N\\big(${S('Enc')}(\\y_i),\\ ${S('h')}^2 I\\big)`,
    after: ['A model in latent space makes small errors, and the decoder decides where those errors land in data space. When the decoder is continuous, a small error in z is a small error in x. When it is not, the same small error can become a large one. Reconstruction cannot see this, because it only evaluates the decoder on the latents of real data. The tokenizers that generate well are those whose latents are spatially coherent and locally continuous, and some recent autoencoders are trained with a diffusion prior so that their latents are easy to model.'],
    check: {
      q: 'The decoder of the strip encoder is exact. Why are its decoded samples worse than those of the identity encoder?',
      opts: [['The model smooths across strip boundaries, and the decoder sends those latents to distant places', true], ['The strip encoder loses information about the data', false], ['The model in latent space is trained on fewer points', false]],
      fb: { true: 'Right. Nothing is lost in the encoding, and the model is the same. Its samples spread a distance of about h around each training latent, and near a cut the decoder maps that spread to two places that are a whole strip apart in data space.', false: 'The encoder is invertible and both models use the same 400 training points. The difference is what happens to the model\'s small errors. Near a cut, a latent that is h away from a training latent can decode to a point in a different strip, far from the data.' },
    },
  },
  {
    id: 'repa', kicker: 'Using a pretrained encoder', title: 'Representation alignment through Tweedie', scene: 'repa',
    prose: [
      'REPA adds an auxiliary loss to diffusion training. A small projection \\(g_\\phi\\) of an intermediate hidden state of the denoiser is trained to match the features \\(\\bm f(\\x_1)\\) that a frozen encoder computes on the clean image. With a squared error in place of the cosine similarity used in practice, this is a regression of \\(\\bm f(\\x_1)\\) on \\(\\x_t\\), and its minimizer is the conditional mean in the formula: the Tweedie estimate of the features.',
      'The data are the 2D mixture, and the frozen encoder is the Bayes classifier \\(\\bm f(\\x_1) = P(k\\mid\\x_1)\\), the probabilities of the three classes. For this encoder the REPA target is the class posterior given the noisy point. Drag the ring \\(\\x_t\\) and move \\(t\\). The filled bars are the target, and the outlined bars are \\(\\bm f(\\hx_1)\\), the features of the denoised point. The plot on the right shows how much of the variance of \\(\\bm f(\\x_1)\\) each of the two explains along the path.',
    ],
    tex: `${S('hid')}(\\x, ${t}) \\;\\to\\; \\E\\big[${feat}(\\x_1)\\,\\big|\\,${x} = \\x\\big] = ${Pk} \\;\\ne\\; ${feat}\\big(${hx}(\\x, ${t})\\big)`,
    deriv: {
      have: ['The regression lemma of §2.2: the minimizer of \\(\\E\\|g(\\x_t) - Y\\|^2\\) over all functions g is \\(\\E[Y\\mid\\x_t]\\).', 'The class k, the clean point \\(\\x_1\\) and the noisy point \\(\\x_t\\) form a chain: \\(\\x_t\\) depends on k only through \\(\\x_1\\).'],
      goal: `\\E[${feat}(\\x_1)\\mid\\x_t] = P(k\\mid\\x_t)`,
      plan: 'Apply the regression lemma with \\(Y = \\bm f(\\x_1)\\), then write \\(\\bm f\\) as a conditional expectation and use the tower property.',
      title: 'Why the REPA target is the noisy class posterior', lines: [
        { tex: `\\arg\\min_g \\E\\big\\|g(\\x_t) - ${feat}(\\x_1)\\big\\|^2 = ${CHG('\\E[\\bm f(\\x_1)\\mid\\x_t]')}`, why: 'The regression lemma with the features of the clean image as the label.', uses: ['regression-lemma'] },
        { tex: `${feat}(\\x_1)_k = P(k\\mid\\x_1) = ${CHG('\\E[\\mathbf 1\\{K = k\\}\\mid\\x_1]')}`, why: 'A probability is the conditional expectation of an indicator.', uses: ['cond-exp'] },
        { tex: `\\E\\big[\\E[\\mathbf 1\\{K = k\\}\\mid\\x_1]\\,\\big|\\,\\x_t\\big] = \\E[\\mathbf 1\\{K = k\\}\\mid\\x_t] = ${CHG('P(k\\mid\\x_t)')}`, why: 'Given \\(\\x_1\\), the noisy point carries no further information about k, so conditioning on \\(\\x_1\\) and then on \\(\\x_t\\) is the same as conditioning on \\(\\x_t\\) alone (the tower property).', uses: ['cond-exp'] },
      ],
    },
    after: ['For a linear encoder the conditional mean passes through, and the target would be \\(\\bm f(\\hx_1)\\): an early layer would be asked to have computed a linear image of the final prediction. For a nonlinear encoder the target is the posterior mean in feature space, which is a different and better-calibrated quantity. The outlined bars show the difference. The denoised point \\(\\hx_1\\) is an average and can sit between the classes, yet the encoder assigns it confidently to one class, so \\(\\bm f(\\hx_1)\\) explains less of the variance than the target does, and at small t less than a constant. REPA therefore supplies an intermediate target that the network can reach early, on the way to the final prediction.'],
    check: {
      q: 'Suppose the frozen encoder were linear, \\(\\bm f(\\x_1) = A\\x_1\\). What would the REPA target be?',
      opts: [['\\(A\\hx_1\\), the encoder applied to the denoised point', true], ['The class posterior \\(P(k\\mid\\x_t)\\)', false], ['\\(A\\x_t\\), the encoder applied to the noisy point', false]],
      fb: { true: 'Right. A conditional expectation is linear, so \\(\\E[A\\x_1\\mid\\x_t] = A\\,\\E[\\x_1\\mid\\x_t] = A\\hx_1\\). Only for a nonlinear encoder do the filled and outlined bars differ.', false: 'The REPA target is always \\(\\E[\\bm f(\\x_1)\\mid\\x_t]\\). For \\(\\bm f(\\x_1) = A\\x_1\\) the matrix moves out of the expectation, which gives \\(A\\hx_1\\).' },
    },
  },
  {
    id: 'pixels', kicker: 'Choosing what to predict', title: 'Back to pixels: which target a narrow network can learn', scene: 'pixels',
    prose: [
      'The prediction targets of §2.2, data \\(\\x_1\\), noise \\(\\x_0\\) and velocity \\(\\vel = \\x_1 - \\x_0\\), have the same minimizer once converted to \\(\\hx_1\\). They differ in which function the network has to represent. Li and He (2025) argue that clean images lie near a low-dimensional set while noise does not, so a network of limited width can predict \\(\\x_1\\) for a large patch but cannot predict \\(\\x_0\\) or \\(\\vel\\). This step computes the argument exactly for a linear network.',
      'The eight digits from §2.7 are the data, with \\(D = 144\\) pixels each. Their covariance has rank 7, so seven directions carry variance and 137 carry none. For each target the page computes the best linear predictor that passes through \\(r\\) dimensions and converts its prediction to \\(\\hx_1\\). Move \\(r\\) and \\(t\\). The \\(\\x_1\\) predictor reaches its best error at \\(r = 7\\), while the \\(\\x_0\\) predictor needs every one of the 144 dimensions.',
    ],
    tex: `\\E\\|${hx} - \\x_1\\|^2 = (${D} - ${r})\\,\\frac{${sg}^2}{${al}^2} \\quad\\text{for the } ${x0}\\text{ predictor of width } ${r} \\le ${D} - 7`,
    deriv: {
      have: ['In the eigenbasis of the data covariance the coordinates of \\(\\x_1 - \\bm m\\) are uncorrelated with variances \\(\\lambda_j\\), and those of \\(\\x_0\\) are uncorrelated with variance 1.', 'A linear predictor of width r keeps the r directions whose fitted values have the largest variance (reduced-rank regression).'],
      goal: '\\text{each dropped direction costs } \\sigma_t^2/\\alpha_t^2 \\text{ for the } \\x_0 \\text{ predictor}',
      plan: 'Write one coordinate as a scalar regression, find which directions the \\(\\x_0\\) predictor keeps, and compute the error of \\(\\hx_1\\) along a dropped one.',
      title: 'Why a narrow noise predictor fails', lines: [
        { tex: `\\xi_j = \\alpha_t a_j + \\sigma_t n_j, \\qquad \\operatorname{Var}\\xi_j = \\alpha_t^2${lam} + \\sigma_t^2, \\qquad \\operatorname{Cov}(\\xi_j, n_j) = ${CHG('\\sigma_t')}`, why: 'Project \\(\\x_t - \\alpha_t\\bm m\\) on the j-th direction. Here \\(a_j\\) is the coordinate of the clean image and \\(n_j\\) that of the noise.', uses: ['expand-square'] },
        { tex: `\\frac{\\operatorname{Cov}^2}{\\operatorname{Var}} = \\frac{\\sigma_t^2}{\\alpha_t^2\\lambda_j + \\sigma_t^2} = ${CHG('1')} \\ \\text{ when } \\lambda_j = 0`, why: 'This is the variance of the fitted values in direction j. It is largest in the 137 directions without data variance, so a narrow \\(\\x_0\\) predictor spends its width there first.', uses: ['expectation'] },
        { tex: `\\hat n_j = 0 \\;\\Rightarrow\\; \\hat a_j = \\frac{\\xi_j}{\\alpha_t} = a_j + \\frac{\\sigma_t}{\\alpha_t}n_j \\;\\Rightarrow\\; \\E(\\hat a_j - a_j)^2 = ${CHG('\\frac{\\sigma_t^2}{\\alpha_t^2}')}`, why: 'In a dropped direction the predicted noise is zero, and converting to data prediction with \\(\\hx_1 = (\\x_t - \\sigma_t\\hx_0)/\\alpha_t\\) leaves the noise in place, whatever \\(\\lambda_j\\) is.', uses: ['substitution'] },
      ],
    },
    after: ['The x<sub>0</sub> and velocity predictors have to pass the noise through in every direction, because the noise lives in all D dimensions and the target contains it. The data predictor only has to output something in the seven directions in which the data vary, and can set the rest to the mean. With large patches D is large, which is why the authors pair data prediction with large patches on raw pixels. The page uses a linear model with a free bias, so the numbers are exact; the authors\' argument concerns trained nonlinear networks.'],
    check: {
      q: 'What is the smallest width r at which the \\(\\x_1\\) predictor reaches its best error, for any t?',
      answer: 7, tol: 0,
      fb: { true: 'Right. The data covariance has rank 7, so outside those seven directions the best prediction of \\(\\x_1\\) is the mean, which needs no width. Set r to 7 and compare its error with r = 144.', false: 'The clean digits vary in only seven directions, the rank of their covariance. In every other direction the best prediction of \\(\\x_1\\) is the constant mean, so seven dimensions suffice.' },
    },
  },
  {
    id: 'attention', kicker: 'Backbones', title: 'Attention is the empirical denoiser', scene: 'attention',
    prose: [
      'The diffusion transformer replaces the convolutional U-Net with a standard transformer on the patch tokens. The step from §2.7 to attention is short. The empirical denoiser is a softmax-weighted average of the training points, and attention is a softmax-weighted average of values. Expanding the squared distance shows that the two are the same computation, with the query \\(\\bm q = \\x\\), the keys \\(\\bm k_i = \\alpha_t\\y_i\\), the values \\(\\bm v_i = \\y_i\\), the temperature \\(\\tau = \\sigma_t^2\\) and one bias per key.',
      'Drag the ring \\(\\x_t\\) and move \\(t\\). The bars show the weights of the 24 training points computed both ways, as dot-product attention and as the distance softmax of §2.7, drawn on top of each other. The readout shows the largest difference, which is rounding error.',
    ],
    tex: `${hx} = \\sum_i \\operatorname{softmax}_i\\Big(\\frac{\\langle ${q}, ${key}\\rangle}{${tau}} + b_i\\Big)\\,${val}, \\qquad b_i = -\\frac{${al}^2\\|${yi}\\|^2}{2${sg}^2}`,
    deriv: {
      have: ['The empirical denoiser of §2.7: \\(\\hx_1 = \\sum_i w_i\\y_i\\) with \\(w_i = \\operatorname{softmax}_i(-\\|\\x - \\alpha_t\\y_i\\|^2/2\\sigma_t^2)\\).', 'A softmax does not change when the same number is added to every logit.'],
      goal: `w_i = \\operatorname{softmax}_i\\big(\\langle\\x, \\alpha_t\\y_i\\rangle/\\sigma_t^2 + b_i\\big)`,
      plan: 'Expand the squared distance and drop the term that is the same for every i.',
      title: 'Why the empirical denoiser is attention', lines: [
        { tex: `-\\frac{\\|\\x - \\alpha_t\\y_i\\|^2}{2\\sigma_t^2} = ${CHG('-\\frac{\\|\\x\\|^2}{2\\sigma_t^2}')} + \\frac{\\langle\\x, \\alpha_t\\y_i\\rangle}{\\sigma_t^2} - \\frac{\\alpha_t^2\\|\\y_i\\|^2}{2\\sigma_t^2}`, why: 'Expand \\(\\|\\bm a - \\bm b\\|^2 = \\|\\bm a\\|^2 - 2\\langle\\bm a, \\bm b\\rangle + \\|\\bm b\\|^2\\).', uses: ['expand-square'] },
        { tex: `w_i = \\operatorname{softmax}_i\\Big(\\frac{\\langle\\x, \\alpha_t\\y_i\\rangle}{\\sigma_t^2} ${CHG('- \\frac{\\alpha_t^2\\|\\y_i\\|^2}{2\\sigma_t^2}')}\\Big)`, why: 'The first term does not depend on i and cancels in the softmax. What remains is a dot product of the query with the key \\(\\alpha_t\\y_i\\), divided by \\(\\sigma_t^2\\), plus a bias per key.', uses: ['log-rules'] },
      ],
    },
    after: ['Time enters only as a scale of the keys and of the temperature. In DiT the time and the class enter through adaptive layer normalization, which regresses a scale and a shift of the activations from the conditioning, and this identity shows why a rescaling is the natural form for time to take. A trained transformer does more than this one layer: it learns the queries, keys and values, and §2.7 showed that what it fails to fit is where generalization comes from.'],
    check: {
      q: 'Why may the term \\(\\|\\x\\|^2\\) be dropped from the logits?',
      opts: [['It is the same for every training point, and a softmax ignores a common shift', true], ['It is small compared with the other terms', false], ['It is zero for normalized inputs', false]],
      fb: { true: 'Right. Adding the same number to every logit multiplies every exponential by the same factor, which cancels in the normalization. The readout shows that the two sets of weights agree to rounding error for any \\(\\x_t\\) and t.', false: 'The term is not small in general. It can be dropped because it does not depend on i, and \\(\\operatorname{softmax}(\\ell_i + c) = \\operatorname{softmax}(\\ell_i)\\) for any constant c.' },
    },
  },
  {
    id: 'mmdit', kicker: 'Text and image tokens', title: 'Joint attention in MMDiT', scene: 'mmdit',
    prose: [
      'A text-to-image model has to let image tokens read the prompt. In the U-Net of Stable Diffusion and in the original DiT for text, image tokens attend to each other and, in a separate cross-attention layer, to the text tokens. The text tokens are never updated by the image. The MMDiT of Stable Diffusion 3 instead concatenates text and image tokens into one sequence and runs a single attention over it, with separate weights for each modality, so that every image token reads every text token and the other way around. Single-stream designs share one set of weights between the modalities after a few modality-specific layers.',
      'The matrix has one row and one column per token, with the text tokens first. Coloured blocks are the pairs that attention computes. Switch between the three designs and move the sliders. The readout gives the number of attention pairs and the number of parameters per layer, counted as \\(12d^2\\) for one set of query, key, value, output and MLP weights of width 4d.',
    ],
    tex: `\\operatorname{Attn}\\big([\\,${S('Nt')} \\text{ text};\\ ${S('Ni')} \\text{ image}\\,]\\big): \\quad (${S('Nt')} + ${S('Ni')})^2 \\text{ pairs}, \\qquad 24\\,${S('d')}^2 \\text{ parameters per MMDiT layer}`,
    after: ['At high resolution the image tokens dominate the sequence, so joint attention costs little more than self-attention over the image alone, and it removes the asymmetry of cross-attention. The single-stream design keeps the same attention pattern with half the weights of MMDiT, which is more parameter-efficient at a fixed size.'],
    check: {
      q: 'Which blocks of the attention matrix does cross-attention leave out?',
      opts: [['Text reading text and text reading the image', true], ['Image reading text', false], ['Image reading the image', false]],
      fb: { true: 'Right. Cross-attention updates only the image tokens: they read each other in self-attention and read the text in cross-attention. The text tokens stay as the text encoder produced them. Joint attention fills the two remaining blocks.', false: 'In a cross-attention model only image tokens are queries. Image reads image and image reads text are computed, but the text tokens are never updated, so the two rows of blocks that start with text are empty.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>The space in which the diffusion runs and the network that parametrizes the field are engineering choices, but several of them have a simple reading through the denoiser: schedules shift with the dimension, latent spaces need continuous decoders, REPA asks for a Tweedie estimate of features, data prediction suits narrow networks on large patches, and attention is the shape of the empirical denoiser. §2.9 turns to data that are not continuous, discrete tokens and text, and §2.11 to the conditioning that the text tokens of this step provide.</p>
      <p>The earlier tutorials, <a href="../score-matching/">Denoising and score matching</a> and <a href="../generalization/">Why diffusion models generalize</a>, derive the denoiser and the empirical optimum used on this page.</p>`,
  },
];
