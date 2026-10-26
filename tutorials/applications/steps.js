// Content of the §7 tutorial: one entry per step. Formulas follow paper/sections/07_applications.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const t = S('t'), x = S('x'), F = S('F'), k = S('k'), eps = S('eps'), W = S('W', '\\|\\W\\|'), sa = S('sa');
const o = S('o'), A = S('A'), H = S('H'), tau = S('tau'), g = S('g'), Phi = S('Phi'), xi = S('xi'), R = S('R'), Om = S('Om');
const y = S('y'), Gt = S('Gt'), kap = S('kap'), M = S('M'), psi = S('psi'), gam = S('gam');

export const INTRO = `
  <div class="kicker">Survey §7 · Interactive tutorial</div>
  <h1>Applications</h1>
  <p>Chapter 2 treated the data as one vector in \\(\\R^d\\), drawn from a distribution of which we have samples, and a generated sample is judged on its own. Images fit this picture, which is why the default recipe of chapter 2, a flow or diffusion model trained with the denoising loss and sampled with an ODE solver, is also the state of the art for images (§7.1). The other applications of chapter 7 each break one part of the picture. In each step below, the default recipe is applied to one such application and fails in a specific way, and the step shows the fix that the field uses.</p>
  <p>The steps come in three groups and a last step. The first group is about what to generate: a robot policy generates actions, for which the average of the data is a failure, and a video generates frames that must belong together. The second is about generating over time: a world model and a weather model roll out on their own output and are judged over the whole rollout. The third is about where the data live: molecules have a rotational symmetry and proteins are made of rigid parts. The last step generates 3D objects from a 2D image model, with a prior but no 3D data at all.</p>
  <p>Steps 2, 5 and 6 run on real data: photographs by Eadweard Muybridge, drug molecules from PubChem and protein backbones from the Protein Data Bank. No network is trained anywhere. Every model is the exact field or conditional of its data, and the world model of step 3 is the exact least-squares fit of a linear model, so every difference on the canvas comes from the design choice and none from training. The convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\); the flow time of the robot policy is written τ.</p>
  <ol class="toc">
    <li><a href="#policy">Robot policies and action chunks</a></li>
    <li><a href="#video">Video: one model for all frames</a></li>
    <li><a href="#world">World models and compounding error</a></li>
    <li><a href="#crps">Weather and the CRPS</a></li>
    <li><a href="#symmetry">Molecules and symmetry</a></li>
    <li><a href="#frames">Protein backbones on SE(3)</a></li>
    <li><a href="#sds">3D from a 2D prior: score distillation</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'policy', kicker: 'What to generate · Actions', title: 'Robot policies and action chunks', scene: 'policy',
    prose: [
      'A policy maps an observation o to the next actions. The default for a continuous output is regression with the squared error, and the derivation shows what it learns: the conditional mean of the actions, which is the denoiser of a flow policy at pure noise. The demonstrations here pass an obstacle on the left or on the right with equal probability, so given the starting position the next actions have two modes, and their mean goes straight through the obstacle. A flow policy learns the whole conditional distribution with the flow-matching loss and samples one mode.',
      'Current policies generate a chunk of H actions at once, execute it and observe again. Both policies here are exact: the regression policy outputs the mean of the next H waypoints given the current position, and the flow policy samples them from their exact conditional. Move H from 1, which replans at every step, to 40, which plans the whole path once.',
    ],
    tex: `\\mathcal L(\\theta) = \\E_{(${o}, ${A}), \\varepsilon, ${tau}}\\big\\|\\vel_\\theta(${A}^{${tau}}, ${tau}; ${o}) - (${A} - \\varepsilon)\\big\\|^2,\\qquad ${A}^{${tau}} = ${tau}\\,${A} + (1 - ${tau})\\,\\varepsilon`,
    deriv: {
      have: ['A regression policy f trained with \\(\\E\\|f(o) - A\\|^2\\) over pairs of observation and action chunk, and a flow policy with \\(A^\\tau = \\tau A + (1 - \\tau)\\varepsilon\\) and the noise ε independent of (o, A).'],
      goal: `f^*(o) = \\E[A\\mid o] = \\E[A\\mid A^0, o]`,
      plan: 'Split the squared error at the conditional mean, then look at the flow policy’s denoiser at τ = 0.',
      title: 'Regression learns the denoiser at pure noise', lines: [
        { tex: `\\E\\big[\\|f(o) - A\\|^2\\mid o\\big] = ${CHG('\\|f(o) - \\E[A\\mid o]\\|^2')} + \\E\\big[\\|A - \\E[A\\mid o]\\|^2\\mid o\\big]`, why: 'Add and subtract \\(\\E[A\\mid o]\\). The cross term has conditional expectation zero.', uses: ['expectation'] },
        { tex: `f^*(o) = ${CHG('\\E[A\\mid o]')}`, why: 'The second term does not depend on f, and the first is zero exactly at the conditional mean.', uses: [] },
        { tex: `A^0 = \\varepsilon\\ \\text{is independent of } A\\quad\\Longrightarrow\\quad \\E[A\\mid A^0, o] = ${CHG('\\E[A\\mid o] = f^*(o)')}`, why: 'At τ = 0 the input of the flow policy is pure noise and carries no information about the chunk, so its denoiser returns the mean of the data.', uses: [] },
        { tex: `A = \\pm b\\ \\text{with probabilities } p, 1 - p\\quad\\Longrightarrow\\quad f^*(o) = ${CHG('(2p - 1)\\,b')}`, why: 'For the two paths around the obstacle, b is the bend to one side. At the start p = 1/2, and the regression policy plans the straight path b = 0 through the obstacle.', uses: [] },
      ],
    },
    after: ['The regression policy passes in 98% of 400 rollouts at H = 1 and in 16% at H = 8; the flow policy passes in 99.5% or more at every H. At H = 1 the regression policy is rescued by feedback: once the modes separate, the conditional mean is very sensitive to the current position, and its own small deviations push it into one mode. With a long chunk it commits to the mean before that happens.', 'The flow policy pays a smaller price for short chunks: at H = 1 it redraws the mode at every step and switches sides 0.36 times per rollout near the obstacle, and from H = 4 on it never does. π0 and π0.5 therefore generate chunks with a flow-matching action expert on top of a vision-language model.'],
    check: {
      q: 'Why does the regression policy fail more often as the chunk gets longer, while the flow policy does not?',
      opts: [
        ['A long chunk is generated from one observation, before the modes have separated. The conditional mean of that chunk is the average of left and right, which crosses the obstacle; a sample of the chunk follows one of the two sides.', true],
        ['Long chunks have more dimensions, and squared-error regression is less accurate in high dimensions.', false],
        ['The regression policy is trained on fewer examples per chunk when H is large.', false],
      ],
      fb: { true: 'Right. The regression policy is exact here, so its failure is not an estimation error: the mean of a bimodal distribution is a point that neither mode contains, and by the derivation it is the flow policy’s output at pure noise, before any sampling has happened.', false: 'Both policies are exact conditionals of the demonstrations, so accuracy is not the issue. Ask what the mean of a left path and a right path looks like.' },
    },
  },
  {
    id: 'video', kicker: 'What to generate · Frames that belong together', title: 'Video: one model for all frames', scene: 'video',
    prose: [
      'A video is a sequence of frames, and the simplest model reuses an image model for every frame. The data here are real: six photographic sequences by Eadweard Muybridge, of a racing, galloping, pacing, walking and jumping horse and a galloping buffalo, reduced to 128 × 80 grey pixels. A training clip is a window of F = 8 consecutive frames, which gives 62 clips built from 104 distinct frames.',
      'No network is trained. Every model is the exact denoiser of its own training set, which for a finite set is a softmax-weighted average of the training items. The joint model treats a clip as one vector of 8 × 2560 pixels and averages whole training clips. The two per-frame models only know the 104 single frames and denoise each frame of a clip on its own, with independent noise per frame or with one noise shared by all frames. Pick a model, move t and press Play to watch the generated clips.',
    ],
    tex: `\\hat{${x}}_1(${x}_{${t}}) = \\sum_n w_n\\,${x}^{(n)},\\qquad w_n \\propto \\exp\\Big(-\\frac{\\|${x}_{${t}} - ${t}\\,${x}^{(n)}\\|^2}{2(1 - ${t})^2}\\Big),\\qquad ${x} = (x^1, \\dots, x^{${F}})`,
    deriv: {
      have: ['A per-frame field \\(\\vel^{\\mathrm f}\\), the exact field of the distribution \\(p^{\\mathrm f}\\) of single frames, applied to every frame \\(x^i\\) of a clip, i = 1, …, F, with the flow map \\(\\phi^{\\mathrm f}\\) that carries N(0, I) to \\(p^{\\mathrm f}\\).'],
      goal: `(x^1_1, \\dots, x^F_1) \\sim \\textstyle\\prod_i p^{\\mathrm f}(x^i)\\ \\text{(independent noise)},\\qquad x^1_1 = \\dots = x^F_1\\ \\text{(shared noise)}`,
      plan: 'Write the ODE of the clip frame by frame and see what each frame depends on.',
      title: 'A per-frame model samples the product of the frame marginals', lines: [
        { tex: `\\tfrac{\\dd}{\\dd t}x^i_t = ${CHG('\\vel^{\\mathrm f}(x^i_t, t)')},\\qquad i = 1, \\dots, F`, why: 'The per-frame model sees only its own frame, so the ODE of the clip splits into F separate ODEs.', uses: [] },
        { tex: `x^i_1 = ${CHG('\\phi^{\\mathrm f}(x^i_0)')}`, why: 'Each frame’s endpoint depends on its own noise and on nothing else.', uses: [] },
        { tex: `x^i_0\\ \\text{independent}\\quad\\Longrightarrow\\quad (x^1_1, \\dots, x^F_1) \\sim ${CHG('\\textstyle\\prod_i p^{\\mathrm f}(x^i)')}`, why: 'Functions of independent variables are independent, and each one has the law \\(p^{\\mathrm f}\\). The frames are drawn independently, so the order of a real clip is lost.', uses: [] },
        { tex: `x^i_0 = x_0\\ \\text{for all } i\\quad\\Longrightarrow\\quad ${CHG('x^1_1 = \\dots = x^F_1')}`, why: 'The same starting point and the same ODE give the same path. Sharing the noise couples the frames completely, and the clip stands still.', uses: [] },
      ],
    },
    after: ['With independent noise every frame is a sharp training frame, but the frames of one clip come from 4.25 different sequences on average and none of the 12 clips is in order. With shared noise all 12 clips are one still image repeated eight times. The joint model plays every clip as smooth motion. Video models therefore denoise all frames of a clip together, with attention across time, and generate long videos chunk by chunk.', 'The right grid names the training frame behind every generated frame, and the joint model only replays its training clips: the exact denoiser of a finite set memorizes it. Why a trained network generates new clips instead is the subject of <a href="../generalization/">Why diffusion models generalize</a> (§2.7).'],
    check: {
      q: 'The per-frame model with shared noise produces clips that stand still, and the one with independent noise produces clips that jump. Which change would make a per-frame model produce clips that move?',
      opts: [
        ['None that keeps it a model of single frames. A per-frame model only knows which frames are likely, not which frame follows which, and that order is a property of the joint distribution.', true],
        ['Sharing the noise only between neighbouring frames, so the clip is smooth and can still change over time.', false],
        ['Training the per-frame model longer, so it learns the single frames more accurately.', false],
      ],
      fb: { true: 'Right. By the derivation, the noise is the only thing that couples the frames of a per-frame model. Partially shared noise interpolates between jumping and standing still, but it has no information about which frame comes next.', false: 'The per-frame denoisers here are already exact. Ask what a model of single frames would need to know to put frame 4 after frame 3.' },
    },
  },
  {
    id: 'world', kicker: 'Generating over time · Compounding error', title: 'World models and compounding error', scene: 'world',
    prose: [
      'A world model generates the next frame from the previous frames and the action of a player, one frame after the other, \\(p(\\x^{1:K}\\mid\\x^0, a^{1:K}) = \\prod_k p(\\x^k\\mid\\x^{\\lt k}, a^k)\\). The default trains each factor on real contexts, which is teacher forcing. Here the game is a strip of 48 pixels with a bright blob, and each action moves the blob left, right or not at all. The model is linear, \\(\\x^{k+1} = [\\x^k, 1]\\,\\W_a\\) with one matrix per action, and it is the exact least-squares fit on 600 real pairs of frames.',
      'After the first frame, the model never sees the game again: it conditions on the frame it generated last, and that frame is not clean. A sampler with a few steps leaves a residual, here pixel noise of standard deviation \\(\\sigma_g\\) on every generated frame. One step multiplies the error of its context by at most \\(\\|\\W\\|\\) and adds a new error, \\(e_k \\le \\|\\W\\|\\,e_{k-1} + \\varepsilon_k\\). Teacher forcing never sees a noisy context, so nothing in its loss keeps \\(\\|\\W\\|\\) small. The second model is trained on contexts with added noise of standard deviation σ. Move both sliders and compare the strips.',
    ],
    tex: `${S('e', `e_{${k}}`)} \\le ${W}\\, ${S('e', `e_{${k}-1}`)} + ${eps}_{${k}},\\qquad \\E_{\\bm{n}}\\big\\|\\y - [${x} + \\bm{n}, 1]\\,\\W\\big\\|^2 = \\big\\|\\y - [${x}, 1]\\,\\W\\big\\|^2 + ${sa}^2\\,\\|\\W_{\\x}\\|_F^2`,
    deriv: {
      have: ['The context noise is \\(\\bm{n} \\sim \\N(0, \\sigma^2 I)\\), independent of the frames. \\(\\W\\) has one row per pixel of the context and a last row for the bias; \\(\\W_{\\x}\\) is \\(\\W\\) without the bias row, so \\([\\bm{n}, 0]\\,\\W = \\bm{n}\\,\\W_{\\x}\\).'],
      goal: `\\E_{\\bm{n}}\\|\\y - [\\x + \\bm{n}, 1]\\,\\W\\|^2 = \\|\\y - [\\x, 1]\\,\\W\\|^2 + \\sigma^2\\|\\W_{\\x}\\|_F^2`,
      plan: 'Split the prediction on the noisy context into the clean prediction and the part that the noise contributes, and expand the square.',
      title: 'Noise in the context is ridge regression', lines: [
        { tex: `\\E\\|\\y - [\\x + \\bm{n}, 1]\\,\\W\\|^2 = \\E\\|${CHG('\\boldsymbol{r} - \\bm{n}\\,\\W_{\\x}')}\\|^2,\\quad \\boldsymbol{r} = \\y - [\\x, 1]\\,\\W`, why: 'The prediction is linear in the context, and the noise does not touch the bias entry.', uses: [] },
        { tex: `= \\|\\boldsymbol{r}\\|^2 - ${CHG('2\\,\\E[\\bm{n}]\\,\\W_{\\x}\\boldsymbol{r}^\\top')} + \\E\\|\\bm{n}\\,\\W_{\\x}\\|^2`, why: 'Expand the square. The residual \\(\\boldsymbol{r}\\) of the clean context does not depend on the noise.', uses: [] },
        { tex: `= \\|\\boldsymbol{r}\\|^2 + ${CHG('\\sigma^2\\,\\mathrm{tr}(\\W_{\\x}^\\top\\W_{\\x})')} = \\|\\boldsymbol{r}\\|^2 + \\sigma^2\\|\\W_{\\x}\\|_F^2`, why: 'The cross term vanishes because \\(\\E[\\bm{n}] = 0\\), and \\(\\E\\|\\bm{n}\\W_{\\x}\\|^2 = \\E[\\bm{n}\\W_{\\x}\\W_{\\x}^\\top\\bm{n}^\\top] = \\sigma^2\\,\\mathrm{tr}(\\W_{\\x}\\W_{\\x}^\\top)\\). Averaged over the training pairs, this is the least-squares loss plus a ridge penalty with \\(\\lambda = \\sigma^2\\) on every weight except the bias.', uses: [] },
      ],
    },
    after: ['At \\(\\sigma_g = 0.01\\), teacher forcing has the smaller one-step error, 0.002 against 0.012 at σ = 0.01, and the larger rollout error, 0.85 against 0.27 after 60 frames, where a clean frame has norm 2.1. Its \\(\\|\\W\\| = 8.7\\): the weights that fit the clean frames best cancel on the data and amplify everything off it. The penalty \\(\\sigma^2\\|\\W_{\\x}\\|_F^2\\) bounds \\(\\|\\W\\|\\), because the spectral norm is at most the Frobenius norm, and noise in the context buys \\(\\|\\W\\| = 1.7\\) for a little one-step accuracy. Too much noise blurs every frame instead: at σ = 0.1 the error after 60 frames is 1.3 whatever \\(\\sigma_g\\). The best σ grows with the residual of the sampler, from 0.002 at \\(\\sigma_g = 0.001\\) to 0.02 at \\(\\sigma_g = 0.02\\), and at its best σ the noisy context is 2 to 3.5 times better than teacher forcing after 60 frames.', 'Both models are fitted to the same data, and only a rollout on generated context tells them apart. World models are therefore trained with corrupted or self-generated contexts: GameNGen adds noise to the context frames, Diffusion Forcing trains with independent noise levels per frame and Self Forcing trains on the model’s own rollouts. For a linear model, the first of these is exactly the ridge penalty of the derivation.'],
    check: {
      q: 'Teacher forcing reaches the smaller one-step error on real frames. Why does it lose the rollout?',
      opts: [
        ['Its loss is computed on clean contexts only, so nothing penalizes large weights. Those weights cancel on real frames but amplify the residual of every generated frame, and a noisy context adds exactly that penalty, σ² times the squared norm of the weights.', true],
        ['A linear model is too small to represent the game, and a larger model trained with teacher forcing would roll out well.', false],
        ['Its one-step error is multiplied at every frame, so the rollout error grows like the one-step error times ‖W‖ to the power k.', false],
      ],
      fb: { true: 'Right. The one-step error measures the model on real contexts, and a rollout runs it on generated ones. Only a loss that sees perturbed contexts controls how the model behaves there.', false: 'Compare the one-step errors and the norms ‖W‖ of the two models in the readout, and ask which of them matters once the context is a generated frame.' },
    },
  },
  {
    id: 'crps', kicker: 'Generating over time · Forecasts are distributions', title: 'Weather and the CRPS', scene: 'crps',
    prose: [
      'A weather forecast is judged as a distribution: what matters is whether the spread of the forecast matches the uncertainty of the weather. The squared error of the ensemble mean, the default score of a deterministic forecast, cannot see the spread at all. GenCast generates an ensemble with a conditional diffusion model, rolled out step by step like a world model, and an ensemble is judged by the continuous ranked probability score. The CRPS rewards members that are close to the observation y and subtracts half the mean distance between members, which rewards spread. It is a proper score: its expectation over observations from the true distribution G is smallest when the forecast is G itself.',
      'The forecast here is the true distribution scaled about its mean by a spread factor κ, so κ \\(\\lt\\) 1 is overconfident and κ = 1 is right. In practice the CRPS is estimated from M members. The plain estimator replaces both expectations by averages over the members; the fair estimator excludes the pairs of a member with itself. Move κ and M, and switch between the two.',
    ],
    tex: `\\mathrm{CRPS}(F, ${y}) = \\E_{X\\sim F}|X - ${y}| - \\tfrac12\\,\\E_{X, X'\\sim F}|X - X'|,\\qquad X = ${kap}\\,Y,\\ \\ Y \\sim ${Gt}`,
    deriv: {
      have: [`M independent members \\(X_1, \\dots, X_M\\) from F and the plain estimator \\(\\frac1M\\sum_i |X_i - y| - \\frac{1}{2M^2}\\sum_{i,j}|X_i - X_j|\\).`],
      goal: `\\E[\\text{plain}] = \\E|X - y| - \\tfrac12\\big(1 - \\tfrac1M\\big)\\E|X - X'|`,
      plan: 'Take the expectation term by term and count the pairs with i = j.',
      title: 'The bias of the plain ensemble estimator', lines: [
        { tex: `\\E\\Big[\\frac1M\\sum_i |X_i - y|\\Big] = ${CHG('\\E|X - y|')}`, why: 'Each member has the law F.', uses: ['expectation'] },
        { tex: `\\E\\Big[\\frac{1}{2M^2}\\sum_{i,j}|X_i - X_j|\\Big] = \\frac{1}{2M^2}\\,${CHG('M(M - 1)')}\\,\\E|X - X'|`, why: 'The M terms with i = j are zero, and the other M(M − 1) pairs are independent copies.', uses: ['expectation'] },
        { tex: `\\E[\\text{plain}] = \\E|X - y| - \\tfrac12\\,${CHG('\\big(1 - \\tfrac1M\\big)')}\\,\\E|X - X'|`, why: 'Combine. The spread term is shrunk by 1 − 1/M, so the estimator rewards spread less than the CRPS does. Dividing by M(M − 1) instead of M² removes the bias; that is the fair estimator.', uses: [] },
      ],
    },
    after: ['The expected CRPS is smallest at κ = 1, with 0.678 against 1.009 for the deterministic forecast at κ = 0, while the squared error of the forecast mean is the same for every κ. The CRPS can tell a calibrated ensemble from an overconfident one, and the squared error cannot.', 'The plain estimator is minimized at κ = 0.76 for M = 5 and at κ = 0.49 for M = 2. Used as a training loss, as in FGN and Swift, it would train a small ensemble to be overconfident; the fair estimator is minimized at κ = 1 for every M.'],
    check: {
      q: 'Why does the plain estimator prefer an overconfident ensemble when M is small?',
      opts: [
        ['Its spread term counts the M zero distances of each member to itself, so it rewards spread by only 1 − 1/M of what the CRPS does, and a narrower ensemble is penalized less than it should be.', true],
        ['With few members the ensemble mean is noisy, and a narrow ensemble reduces that noise.', false],
        ['Because the observation is a single number and an ensemble of M numbers cannot be compared with it fairly.', false],
      ],
      fb: { true: 'Right. The accuracy term is unbiased in both estimators; only the spread term differs. With M = 2 the plain estimator halves the reward for spread and its minimum moves to κ = 0.49.', false: 'Look at the second line of the derivation: which terms of the double sum are zero whatever the members are?' },
    },
  },
  {
    id: 'symmetry', kicker: 'Where the data live · Symmetry', title: 'Molecules and symmetry', scene: 'molecule',
    prose: [
      'A molecule is the same molecule after a rotation, so a generative model of molecules should give \\(\\x\\) and \\(g\\x\\) the same probability. The proposition of §7.6 says how a flow achieves this: if the noise distribution is invariant under the rotations and the velocity is equivariant, every marginal of the flow is invariant. The training data, however, come in whatever orientation the files have. Here each molecule is given in one orientation, and two fields are compared: the plain field, the exact field of these data, and the equivariant field, the best equivariant field for the same data, which is what an equivariant network such as that of EDM learns. The derivation shows that the second is the exact field of the data at all rotations.',
      'The left plot tests equivariance directly. It flows a rotated noise (solid) and rotates the flow of the original noise (dashed rings); if the field is equivariant, the two coincide at every t. The molecules are real 3D conformers from PubChem, hydrogens included: caffeine, aspirin, ibuprofen, paracetamol and nicotine. Translations are removed by centring the molecule, because they have no invariant probability distribution. Pick a molecule and a field, move θ and t, and drag to turn the view.',
    ],
    tex: `\\vel(${g}\\x, ${t}) = ${g}\\,\\vel(\\x, ${t})\\ \\ \\forall ${g}\\quad\\Longrightarrow\\quad ${Phi}(${g}\\x) = ${g}\\,${Phi}(\\x),\\qquad p_{${t}}(${g}B) = p_{${t}}(B)`,
    deriv: {
      have: ['The flow-matching loss \\(\\mathcal L(\\vel; q) = \\E\\|\\vel(\\x_t, t) - (\\x_1 - \\x_0)\\|^2\\) with \\(\\x_1 \\sim q\\), a noise distribution with \\(g_\\#p_0 = p_0\\) for every rotation g, the one-orientation data q and the rotated data \\(\\bar q = \\int g_\\#q\\,\\dd g\\), with the exact field \\(\\bar\\vel\\).'],
      goal: `\\mathcal L(\\vel; q) \\ge \\mathcal L(\\bar\\vel; q)\\ \\text{for every equivariant } \\vel`,
      plan: 'Show that an equivariant field cannot tell q from its rotations, so its loss on q equals its loss on the rotated data.',
      title: 'The best equivariant field is the field of the rotated data', lines: [
        { tex: `\\mathcal L(\\vel; g_\\#q) = \\E\\big\\|\\vel(g\\x_t, t) - g(\\x_1 - \\x_0)\\big\\|^2 = \\E\\big\\|g\\,\\vel(\\x_t, t) - g(\\x_1 - \\x_0)\\big\\|^2 = ${CHG('\\mathcal L(\\vel; q)')}`, why: 'Draw \\(\\x_1\\) from q and rotate it, and write the noise as \\(g\\x_0\\), which has the same law. Then use equivariance, and that a rotation does not change lengths.', uses: [] },
        { tex: `\\mathcal L(\\vel; \\bar q) = \\int \\mathcal L(\\vel; g_\\#q)\\,\\dd g = ${CHG('\\mathcal L(\\vel; q)')}`, why: 'The loss is an expectation over the data, so it is linear in the data distribution.', uses: [] },
        { tex: `\\bar\\vel = \\arg\\min_{\\text{all } \\vel}\\mathcal L(\\vel; \\bar q),\\qquad ${CHG('\\bar\\vel(g\\x, t) = g\\,\\bar\\vel(\\x, t)')}`, why: 'The exact field minimizes the loss over all fields. It is \\(\\E[\\x_1 - \\x_0\\mid\\x_t]\\), and the joint law of noise and rotated data is invariant, so this conditional mean is equivariant.', uses: [] },
        { tex: `\\mathcal L(\\vel; q) = \\mathcal L(\\vel; \\bar q) \\ge \\mathcal L(\\bar\\vel; \\bar q) = ${CHG('\\mathcal L(\\bar\\vel; q)')}`, why: 'Line 2 for \\(\\vel\\), the minimum of line 3, and line 2 again for \\(\\bar\\vel\\), which is equivariant. An equivariant network trained on one orientation learns the same field as a general network trained on randomly rotated copies.', uses: [] },
      ],
    },
    after: ['With the equivariant field, the two molecules on the left agree to within 10⁻¹⁴ Å, which is rounding error, and the 32 generated molecules point in all directions, with a mean rotation from the conformer between 122° and 137° against 126.5° for uniform rotations. With the plain field, the two caffeine molecules end 6.4 Å apart at θ = 115°, and every generated molecule lies within 2° of the one orientation in the data.', 'The derivation gives two routes to the same field: an equivariant network such as the graph network of EDM, or a general network trained on randomly rotated copies of the data, which AlphaFold 3 and SimpleFold use. The second reaches equivariance only approximately, through training, while the first has it by construction.'],
    check: {
      q: 'The noise N(0, I) is invariant under rotations. Why is that not enough for the plain field’s molecules to be invariant?',
      opts: [
        ['The noise only fixes the distribution at t = 0. The field carries it to t = 1, and a field that is not equivariant moves rotated noises to unrotated places, here toward the single orientation it has seen.', true],
        ['Because the noise is invariant only in distribution, not for each sample.', false],
        ['It is enough; the samples look aligned only because 32 samples are too few.', false],
      ],
      fb: { true: 'Right. The proposition needs both an invariant source and an equivariant field. The plain field is the best field for the data it saw, and those data have one orientation.', false: 'The invariance of the marginals is carried from t = 0 to t = 1 by the field. Look at what the proposition needs from the field.' },
    },
  },
  {
    id: 'frames', kicker: 'Where the data live · Rigid parts', title: 'Protein backbones on SE(3)', scene: 'protein',
    prose: [
      'The N, Cα and C atoms of a residue form a nearly rigid triangle, so a backbone of N residues can be generated as N rigid frames: a position \\(\\x_i\\) of the Cα atom and a rotation \\(R_i\\). FrameDiff and FrameFlow generate on this space, \\(\\mathrm{SE}(3)^N\\). The positions move on straight lines, as in every flow in this survey. The rotations cannot: the straight line between two rotation matrices leaves the rotations, and the path follows the geodesic of SO(3) instead.',
      'The targets are real backbones from the Protein Data Bank: ubiquitin (1UBQ, 76 residues), the engrailed homeodomain (1ENH, 54 residues) and the Trp-cage miniprotein (1L2Y, 20 residues). Each residue’s frame comes from its own N, Cα and C atoms. The path starts from Cα positions drawn from a Gaussian of the protein’s size and from uniformly random rotations. The toggle moves the three atoms of every residue on straight lines instead, which is the default recipe applied to atom coordinates. The right plot shows the rotation of one residue, the geodesic as solid axes and the average of the two matrices as dashed ones. Move t or press Play, and drag the protein to turn the view.',
    ],
    tex: `${R}^{${t}} = ${R}^0 \\exp(${t}\\,${Om}),\\quad ${Om} = \\log\\big((${R}^0)^{\\top} ${R}^1\\big),\\qquad \\dot{${R}}^{${t}} = \\frac{${R}^{${t}} \\log\\big((${R}^{${t}})^{\\top} ${R}^1\\big)}{1 - ${t}},\\qquad \\dot{${xi}}^{${t}} = ${xi}^1 - ${xi}^0`,
    deriv: {
      have: ['The geodesic \\(R^t = R^0\\exp(t\\,\\Omega)\\) with \\(\\Omega = \\log((R^0)^\\top R^1)\\), a skew-symmetric matrix.'],
      goal: `\\dot R^t = \\frac{R^t \\log\\big((R^t)^\\top R^1\\big)}{1 - t}`,
      plan: 'Differentiate, then write Ω through the rotation that remains between \\(R^t\\) and \\(R^1\\).',
      title: 'The rotational target velocity', lines: [
        { tex: `\\dot R^t = R^0 \\exp(t\\,\\Omega)\\,\\Omega = ${CHG('R^t\\,\\Omega')}`, why: 'The derivative of \\(\\exp(t\\Omega)\\) is \\(\\exp(t\\Omega)\\Omega\\).', uses: [] },
        { tex: `(R^t)^\\top R^1 = \\exp(-t\\,\\Omega)\\,(R^0)^\\top R^1 = \\exp(-t\\,\\Omega)\\exp(\\Omega) = ${CHG('\\exp\\big((1 - t)\\,\\Omega\\big)')}`, why: 'Rotations are orthogonal, and powers of the same matrix commute.', uses: [] },
        { tex: `\\Omega = \\frac{\\log\\big((R^t)^\\top R^1\\big)}{1 - t}\\quad\\Longrightarrow\\quad \\dot R^t = ${CHG('\\frac{R^t \\log\\big((R^t)^\\top R^1\\big)}{1 - t}')}`, why: 'Take the logarithm, which is valid while the remaining angle (1 − t)θ is below π. This is the rotational analogue of \\((\\x^1 - \\x^t)/(1 - t)\\).', uses: [] },
      ],
    },
    after: ['On the frame path the bond lengths inside every residue are exact at every t. When the atoms move on straight lines, the endpoints are the same, but in between the bonds stretch and shrink: at t = 0.5 the largest bond-length error is 92% for ubiquitin. The model would have to learn the rigid geometry that the frame representation gets for free.', 'For the ringed residue the average of the two matrices at t = 0.5 is not a rotation: its determinant is 0.25. The geodesic stays on the rotations and turns the residue at a constant rate, so the angle left at time t is (1 − t)θ.'],
    check: {
      q: 'Why does the rotational target velocity have the factor 1/(1 − t), like the translational velocity (x¹ − xᵗ)/(1 − t)?',
      opts: [
        ['Both paths move at a constant speed and cover what remains in the time 1 − t that remains. The remaining displacement, x¹ − xᵗ or log((Rᵗ)ᵀR¹), divided by the remaining time is that speed.', true],
        ['The factor compensates for the noise, which has variance 1 − t at time t.', false],
        ['It comes from the Jacobian of the matrix exponential, which is singular at t = 1.', false],
      ],
      fb: { true: 'Right. Both are written without the starting point, as functions of the current state and the target. That is the form a network needs, because at training time it sees \\(R^t\\) and predicts \\(R^1\\).', false: 'Compare \\(\\dot\\x^t = \\x^1 - \\x^0\\) with \\(\\x^1 - \\x^t = (1 - t)(\\x^1 - \\x^0)\\), and the second line of the derivation.' },
    },
  },
  {
    id: 'sds', kicker: 'No data, only a prior', title: '3D from a 2D prior: score distillation', scene: 'sds',
    prose: [
      'Large 3D datasets are much smaller than image datasets, so the first text-to-3D methods had no 3D data to train on, only a pretrained image model. DreamFusion optimizes the parameters ψ of a 3D object so that its renderings look like samples of the image model for a prompt c. Score distillation sampling (SDS) noises a rendering, lets the image model denoise it, and moves ψ toward the denoised image. The derivation shows that this update is the gradient of the paper’s eq. (SDS) and that it moves each object on its own.',
      'Here the image model is an exact mixture in the plane, and the renderer is the identity, so an object is a point and its rendering is the same point. The prompt c selects the two upper components. Each of 200 particles is one object optimized from its own random start. SDS uses the guided denoiser with weight γ. VSD, the variational score distillation of ProlificDreamer, treats the particles as samples of a distribution and subtracts the denoiser of the whole noised cloud. Pick a method and γ, and compare the particles with the samples on the right.',
    ],
    tex: `\\nabla_{${psi}}\\mathcal L_{\\mathrm{SDS}} = \\E_{${t}, \\x_0}\\Big[w(${t})\\big(\\hat\\x_0(\\x_{${t}}, ${t}, c) - \\x_0\\big)\\frac{\\partial g(${psi})}{\\partial ${psi}}\\Big],\\qquad \\hat\\x_1^{${gam}} = \\hat\\x_1(\\x_{${t}}) + ${gam}\\big(\\hat\\x_1(\\x_{${t}}, c) - \\hat\\x_1(\\x_{${t}})\\big)`,
    deriv: {
      have: ['The identity renderer \\(g(\\psi) = \\psi\\), the noised rendering \\(\\x_t = t\\,\\psi + (1 - t)\\,\\x_0\\), the model’s denoiser \\(D(\\x_t) = \\hat\\x_1^\\gamma(\\x_t)\\) and its noise prediction \\(\\hat\\x_0 = (\\x_t - t\\,D(\\x_t))/(1 - t)\\).'],
      goal: `\\psi \\leftarrow \\psi + \\lambda\\big(D(\\x_t) - \\psi\\big)`,
      plan: 'Express both noises through the denoisers, then take one gradient step.',
      title: 'SDS moves each object toward its own denoised version', lines: [
        { tex: `\\x_0 = ${CHG('\\frac{\\x_t - t\\,\\psi}{1 - t}')}`, why: 'Solve the definition of \\(\\x_t\\) for the noise. This is the noise prediction of a point mass at ψ.', uses: [] },
        { tex: `\\hat\\x_0 - \\x_0 = ${CHG('-\\frac{t}{1 - t}\\big(D(\\x_t) - \\psi\\big)')}`, why: 'Subtract from the model’s noise prediction; the \\(\\x_t\\) terms cancel.', uses: [] },
        { tex: `\\psi \\leftarrow \\psi - \\lambda\\,w(t)\\,(\\hat\\x_0 - \\x_0) = ${CHG('\\psi + \\lambda\\big(D(\\x_t) - \\psi\\big)')}`, why: 'A gradient step with \\(\\partial g/\\partial\\psi = I\\) and the weight \\(w(t) = (1 - t)/t\\). The update of one particle does not depend on the others, so all particles solve the same problem.', uses: [] },
        { tex: `\\text{VSD:}\\quad \\psi \\leftarrow \\psi + \\lambda\\big(D(\\x_t) - ${CHG('D_q(\\x_t)')}\\big)`, why: 'VSD replaces the point mass at ψ by the distribution q of the particles. Its denoiser \\(D_q\\) is exact here; in ProlificDreamer it is a second network trained on the renderings. This is the update of distribution matching distillation.', uses: [] },
      ],
    },
    after: ['With SDS at γ = 1 all particles end in one clump near (−0.71, 1.06), between the two modes of the prompt, and only 22% lie within 2.5 standard deviations of a mode. At high noise the denoiser returns the mean of the prompt, and SDS averages over all noise levels. Guidance pulls the clump into a mode and then past it: at γ = 7.5 it sits at (−1.66, 2.55) and no particle lies in the data. This is the oversaturation of SDS, and DreamFusion uses a guidance weight of 100.', 'VSD at γ = 1 puts 56% and 45% of the particles in the two modes, against weights of 57% and 43%, and 95% within 2.5 standard deviations, against 94% for exact samples. Subtracting the denoiser of the cloud pushes particles away from each other where the cloud is denser than the model. The same correction turns score distillation into the distribution matching of <a href="../distillation/">Distillation</a> (§5).'],
    check: {
      q: 'Why do the SDS particles end in one clump although they start at different points?',
      opts: [
        ['Each particle follows the same update, toward the denoised version of itself, and nothing in the update depends on the other particles. Particles that start in the same basin reach the same point.', true],
        ['The learning rate is too large, so the particles overshoot into the same place.', false],
        ['The image model has only one mode for the prompt, so every optimizer must find it.', false],
      ],
      fb: { true: 'Right. SDS optimizes 200 copies of one problem. Diversity needs a term that sees the other particles, and the denoiser of the cloud in VSD is that term.', false: 'The prompt has two modes, and the right plot shows samples in both. Look at the third line of the derivation: what does the update of one particle depend on?' },
    },
    closing: `<h3>Where this leads</h3>
      <p>Each step broke one part of the picture of chapter 2, a single vector judged on its own and drawn from a distribution of which we have samples. The table collects what broke, how the default recipe failed and what fixed it, with the numbers from the text.</p>
      <table class="tbl"><thead><tr><th>Application</th><th>What breaks</th><th>How the default fails</th><th>The fix</th></tr></thead><tbody>
        <tr><td>1. Policy</td><td>The mean of the data is not a valid output</td><td>Regression plans the mean path into the obstacle and passes in 16% at H = 8</td><td>A flow policy over action chunks: 99.8%</td></tr>
        <tr><td>2. Video</td><td>The frames are not independent</td><td>Per-frame denoising samples the product of the frame marginals: 0 of 12 clips in order</td><td>Denoise all frames of a clip jointly</td></tr>
        <tr><td>3. World model</td><td>The model conditions on its own output</td><td>Teacher forcing never sees a noisy context: its weights amplify the residual, 2 to 3.5 times the error of a noisy-context model after 60 frames</td><td>Train on corrupted or self-generated context</td></tr>
        <tr><td>4. Weather</td><td>The output is judged as a distribution</td><td>The squared error ignores the spread, and the plain CRPS estimate prefers κ = 0.76 at M = 5</td><td>The fair CRPS, minimized at κ = 1</td></tr>
        <tr><td>5. Molecule</td><td>Rotations do not change the object</td><td>The plain field generates only the orientation of the data</td><td>An equivariant network or rotated copies, which give the same field</td></tr>
        <tr><td>6. Protein</td><td>The data contain rigid parts</td><td>Straight atom paths stretch bonds by 92% at t = 0.5</td><td>Frames on SE(3) with geodesic rotations</td></tr>
        <tr><td>7. 3D</td><td>There are no data, only a 2D prior</td><td>SDS collapses to one point and leaves the data under guidance</td><td>VSD, which matches the distribution of the particles</td></tr>
      </tbody></table>
      <p>Images are the row the table leaves out: nothing in the picture breaks, and the default recipe is the state of the art (§7.1). Chapter 7 also covers applications that this page leaves out. Crystals live on a torus, so the noise is a wrapped Gaussian and paths take the shortest way modulo one. Medical imaging is mostly an inverse problem, solved with the posterior samplers of <a href="../inference-time/">Inference-time scaling</a> (§6). Audio and video are generated jointly by models that share attention across both streams. The fields behind every step here are those of <a href="../flow-matching/">Flow matching</a> (§2.4) and <a href="../couplings/">Couplings, transport and control</a> (§2.6); a guided or reward-tuned version of each model is the subject of <a href="../guidance/">Conditioning and guidance</a> (§2.11) and <a href="../post-training/">Reward fine-tuning</a> (§4), and the few-step generators that make real-time world models and policies possible are those of <a href="../distillation/">Distillation</a> (§5). Animations <a href="../../animations.html#a12">A12, self forcing</a>, <a href="../../animations.html#a15">A15, an action-conditioned world model</a>, and <a href="../../animations.html#a16">A16, protein frames on SE(3)</a>, show steps 3 and 6 at a larger scale.</p>`,
  },
];
