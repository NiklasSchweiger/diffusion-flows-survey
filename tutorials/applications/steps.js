// Content of the §7 tutorial: one entry per step. Formulas follow paper/sections/07_applications.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const t = S('t'), x = S('x'), F = S('F'), tf = S('tf'), k = S('k'), a = S('a'), eps = S('eps'), K = S('K'), e = S('e'), L = S('L');
const o = S('o'), A = S('A'), H = S('H'), tau = S('tau'), g = S('g'), Phi = S('Phi'), xi = S('xi'), R = S('R'), Om = S('Om');
const y = S('y'), Gt = S('Gt'), kap = S('kap'), M = S('M');

export const INTRO = `
  <div class="kicker">Survey §7 · Interactive tutorial</div>
  <h1>Applications</h1>
  <p>Images are the easy case for a generative model: one sample, no symmetry to respect and no consequences once it is drawn. The applications of chapter 7 add structure that the model has to get right. A video is many frames that must agree with each other. A world model and a robot policy generate step by step, and each step conditions on what the model produced before. Molecules and proteins have symmetries and rigid parts, and a weather forecast is judged as a distribution rather than as a single picture.</p>
  <p>Each step isolates one of these issues in a toy whose field or conditional is known exactly, so every difference on the canvas comes from the design choice and none from training. Steps 1, 2, 4, 5 and 6 are exact up to the ODE solver. Step 3 is Monte Carlo over seeded rollouts of an exact conditional. The convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\); the flow time of the robot policy is written τ.</p>
  <ol class="toc">
    <li><a href="#video">Video: one model for all frames</a></li>
    <li><a href="#world">World models and compounding error</a></li>
    <li><a href="#policy">Robot policies and action chunks</a></li>
    <li><a href="#symmetry">Molecules and symmetry</a></li>
    <li><a href="#frames">Protein backbones on SE(3)</a></li>
    <li><a href="#crps">Weather and the CRPS</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'video', kicker: 'Frames that belong together', title: 'Video: one model for all frames', scene: 'video',
    prose: [
      'A clip is not a set of independent images. Here a clip is a dot that moves across 12 frames at a constant speed of 1.5, to the left or to the right, from a random starting offset. Each single frame looks the same whichever way the dot moves, so a model of single frames sees a blurred two-mode distribution and nothing that tells it about motion.',
      'The joint model generates all F frames as one vector, with the exact velocity of the clip distribution. The two per-frame models use the exact flow of each frame on its own, which is the best a model of single frames can do, and differ only in their noise: each frame gets its own noise, or all frames share one. Switch between the four and move t to watch the clips form.',
    ],
    tex: `${x} = (x^1, \\dots, x^{${F}}),\\qquad x^f = c + 1.5\\,s\\,${tf} + \\eta_f,\\qquad \\vel_${t}(${x}) \\ne \\big(\\vel^{(1)}_${t}(x^1), \\dots, \\vel^{(${F})}_${t}(x^{${F}})\\big)`,
    after: ['The joint model reproduces the data: the jitter, the root mean square of the second difference along the clip, is 0.119 against 0.115 in the data, and every one of the 400 clips moves with a speed within 0.5 of ±1.5. With independent noise per frame, every frame on its own is a correct sample, but the clip flickers with a jitter of 1.724 and only 20% of the clips have a plausible speed. With shared noise the flicker is gone (jitter 0.022), but the same noise maps to the same quantile of every frame. The frames differ only in how far apart their two modes are, so each clip bends symmetrically about the middle frame and its fitted speed is 0. No clip travels.', 'Both failures have the same cause. The coupling between frames is a property of the joint distribution, and it cannot be recovered from the marginals of the frames, whatever noise the frames share. Video models therefore denoise all frames of a clip together, with attention across time, and long videos are generated chunk by chunk, each chunk conditioned on the previous one. That causal factorization is the subject of the next step.'],
    check: {
      q: 'The per-frame model with shared noise produces smooth clips that do not travel. Which change would make it produce moving clips?',
      opts: [
        ['None that keeps it a model of single frames. Every per-frame model with any shared or independent noise only reproduces the marginal of each frame, and the speed is a property of the joint distribution.', true],
        ['Sharing the noise only between neighbouring frames, so the clip is smooth and can still change over time.', false],
        ['Training the per-frame model longer, so its marginals become more accurate.', false],
      ],
      fb: { true: 'Right. The joint model differs from the per-frame models only in its field, which looks at all frames at once. Partially shared noise interpolates between flicker and standing still, but it has no information about which way the dot moves.', false: 'The marginals here are already exact. Ask what the per-frame model would need to know to decide whether the dot moves left or right.' },
    },
  },
  {
    id: 'world', kicker: 'Generating on its own output', title: 'World models and compounding error', scene: 'world',
    prose: [
      'A world model generates the next frame from the previous frames and the action of a player or an agent, one frame after the other, \\(p(\\x^{1:K}\\mid\\x^0, a^{1:K}) = \\prod_k p(\\x^k\\mid\\x^{<k}, a^k)\\). Here the world is a ring of 32 cells with one bright cell, and each key moves it one cell to the left, to the right or not at all. The model is specified by hand: it moves a share 1 − ε of the intensity correctly and spills the rest one cell too far or too short.',
      'Model B differs from model A only off the data: before it steps, it snaps its context to the brightest cell, which leaves every real frame unchanged. Both therefore have the same one-step error ε on real contexts, the quantity that teacher-forced training minimizes. Run both on their own output and compare.',
    ],
    tex: `\\mathrm{TV}\\big(p(\\x^{1:${K}}), q(\\x^{1:${K}})\\big) \\le \\sum_{${k}=1}^{${K}} ${eps}_${k},\\qquad ${e}_${K} \\le ${eps}\\,\\frac{${L}^${K} - 1}{${L} - 1}`,
    deriv: {
      have: [`The error of the generated frame grows by at most a factor L in one step and gains at most ε: \\(e_k \\le L\\,e_{k-1} + \\varepsilon\\), with \\(e_0 = 0\\) because the first frame is real.`],
      goal: `e_K \\le \\varepsilon\\,\\frac{L^K - 1}{L - 1}`,
      plan: 'Unroll the recursion from k = K down to k = 0.',
      title: 'The drift of a Lipschitz rollout', lines: [
        { tex: `e_K \\le L\\,e_{K-1} + \\varepsilon \\le L\\big(${CHG('L\\,e_{K-2} + \\varepsilon')}\\big) + \\varepsilon`, why: 'Apply the recursion to \\(e_{K-1}\\). The bound is monotone in its argument because L ≥ 0.', uses: [] },
        { tex: `\\le L^K e_0 + \\varepsilon\\,${CHG('(1 + L + \\dots + L^{K-1})')}`, why: 'Repeat K times.', uses: [] },
        { tex: `= \\varepsilon\\,${CHG('\\frac{L^K - 1}{L - 1}')}`, why: 'The first term is zero, and the second is a geometric sum. For L = 1 it is Kε, the linear growth of the TV bound; for L < 1 it stays below ε/(1 − L) for every K.', uses: [] },
      ],
    },
    after: ['At ε = 0.05 the error of model A is 0.311 after 8 frames, 0.653 after 32 and 0.768 after 64. The smeared intensity is never corrected, so A behaves like L = 1, and its error comes close to the linear bound Kε for the first frames. The bound itself is vacuous from frame 20 on. Model B has an error of exactly 0.05 at every frame: it reads its own smeared frame as the nearest valid frame, so its errors do not accumulate (L = 0 near the data). This holds for every ε below 0.625, where the correct cell stops being the brightest.', 'Both models have the same teacher-forced loss, and only a rollout on generated context tells them apart. This is why world models are trained with corrupted or self-generated contexts: GameNGen adds noise to the context frames, Diffusion Forcing trains with independent noise levels per frame and Self Forcing trains on the model’s own rollouts. Each of these teaches the model what to do with a context that is slightly off.'],
    check: {
      q: 'The TV bound of the proposition holds for both models and is the same for both. Why does it not distinguish them?',
      opts: [
        ['The bound uses only the one-step errors on real contexts, which are equal. It says nothing about what a model does on its own, slightly wrong, frames, and that is where A and B differ.', true],
        ['The bound is too loose by a constant factor; a tighter constant would separate the two models.', false],
        ['The bound applies only to stochastic models, and both models here are deterministic.', false],
      ],
      fb: { true: 'Right. The proposition guarantees that a small teacher-forced error cannot hurt more than linearly, and model A almost reaches that worst case. Model B shows that a model can do much better than the bound, which takes a property of the model off the data, such as a Lipschitz constant below 1.', false: 'Look at which contexts the errors \\(\\varepsilon_k\\) are measured on, and which contexts the models see during a rollout.' },
    },
  },
  {
    id: 'policy', kicker: 'Actions are multimodal', title: 'Robot policies and action chunks', scene: 'policy',
    prose: [
      'A policy maps an observation o to the next actions. The demonstrations here pass an obstacle on the left or on the right with equal probability, so given the starting position the next actions have two modes. A regression policy trained with the squared error learns their conditional mean, which is the path straight through the obstacle. A flow policy learns the whole conditional distribution with the flow-matching loss and samples one mode.',
      'Current policies generate a chunk of H actions at once, execute it and observe again. Both policies here are exact: the regression policy outputs the mean of the next H waypoints given the current position, and the flow policy samples them from their exact conditional. Move H from 1, which replans at every step, to 40, which plans the whole path once.',
    ],
    tex: `\\mathcal L(\\theta) = \\E_{(${o}, ${A}), \\varepsilon, ${tau}}\\big\\|\\vel_\\theta(${A}^{${tau}}, ${tau}; ${o}) - (${A} - \\varepsilon)\\big\\|^2,\\qquad ${A}^{${tau}} = ${tau}\\,${A} + (1 - ${tau})\\,\\varepsilon`,
    after: ['The flow policy passes in 100% of 400 rollouts at every H up to 4 and in 99.5% or more at H = 8, 16 and 40. The regression policy passes in 98% at H = 1, 88% at H = 2, 67% at H = 4, 16% at H = 8, 3% at H = 16 and never at H = 40. At H = 40 it executes the mean of the two modes, which goes straight into the obstacle. At H = 1 it is rescued by feedback: as the two modes separate, the conditional mean becomes very sensitive to the current position, a small deviation to one side pulls the next mean further to that side, and the regression policy is pushed into one mode by its own noise. That works only with a fresh observation at almost every step.', 'Short chunks have a cost for the flow policy too. Each replanning draws the mode again from its posterior, and at H = 1 the policy switches between left and right 0.36 times per rollout near the obstacle; at H = 2 it is 0.09 and from H = 4 on it is zero. Committing to a sampled chunk removes this dithering. π0 and π0.5 generate chunks with a flow-matching action expert on top of a vision-language model.'],
    check: {
      q: 'Why does the regression policy fail more often as the chunk gets longer, while the flow policy does not?',
      opts: [
        ['A long chunk is generated from one observation, before the modes have separated. The conditional mean of that chunk is the average of left and right, which crosses the obstacle; a sample of the chunk follows one of the two sides.', true],
        ['Long chunks have more dimensions, and squared-error regression is less accurate in high dimensions.', false],
        ['The regression policy is trained on fewer examples per chunk when H is large.', false],
      ],
      fb: { true: 'Right. The regression policy is exact here, so its failure is not an estimation error: the mean of a bimodal distribution is a point that neither mode contains. This is the same quantity as the denoiser at pure noise, the average of the data.', false: 'Both policies are exact conditionals of the demonstrations, so accuracy is not the issue. Ask what the mean of a left path and a right path looks like.' },
    },
  },
  {
    id: 'symmetry', kicker: 'Rotating a molecule changes nothing', title: 'Molecules and symmetry', scene: 'molecule',
    prose: [
      'A molecule is the same molecule after a rotation, so a generative model of molecules should give \\(\\x\\) and \\(g\\x\\) the same probability. The proposition of §7.6 says how a flow achieves this: if the noise distribution is invariant under the rotations and the velocity is equivariant, every marginal of the flow is invariant. The left plot tests the equivariance directly. It flows a rotated noise (solid) and rotates the flow of the original noise (dashed); if the field is equivariant, the two coincide at every t.',
      'The molecule is a water molecule in the plane, with its centre fixed at the origin; translations have no invariant probability distribution, so they are removed. The data contain it at all orientations, and the field is the exact velocity of that distribution. Move θ and t. Then train on a single orientation and compare.',
    ],
    tex: `\\vel(${g}\\x, ${t}) = ${g}\\,\\vel(\\x, ${t})\\ \\ \\forall ${g}\\quad\\Longrightarrow\\quad ${Phi}(${g}\\x) = ${g}\\,${Phi}(\\x),\\qquad p_${t}(${g}B) = p_${t}(B)`,
    deriv: {
      have: ['A velocity with \\(\\vel(g\\x, t) = g\\,\\vel(\\x, t)\\) for a rotation g, and the flow map \\(\\phi_t\\) with \\(\\frac{\\dd}{\\dd t}\\phi_t(\\x) = \\vel(\\phi_t(\\x), t)\\) and \\(\\phi_0(\\x) = \\x\\).'],
      goal: `\\phi_t(g\\x) = g\\,\\phi_t(\\x)`,
      plan: 'Show that the rotated path solves the same ODE from the rotated starting point, then use uniqueness.',
      title: 'An equivariant field gives an equivariant flow', lines: [
        { tex: `\\tfrac{\\dd}{\\dd t}\\,g\\,\\phi_t(\\x) = g\\,${CHG('\\vel(\\phi_t(\\x), t)')}`, why: 'g is a fixed linear map, so it commutes with the time derivative.', uses: [] },
        { tex: `= ${CHG('\\vel(g\\,\\phi_t(\\x), t)')}`, why: 'Equivariance of the field.', uses: [] },
        { tex: `g\\,\\phi_0(\\x) = g\\x\\quad\\Longrightarrow\\quad ${CHG('g\\,\\phi_t(\\x) = \\phi_t(g\\x)')}`, why: 'The curve \\(g\\phi_t(\\x)\\) starts at gx and solves the ODE, and so does \\(\\phi_t(g\\x)\\). Solutions of an ODE with a Lipschitz field are unique.', uses: [] },
        { tex: `p_t(gB) = p_0\\big(\\phi_t^{-1}(gB)\\big) = p_0\\big(g\\,\\phi_t^{-1}(B)\\big) = ${CHG('p_t(B)')}`, why: 'The marginal is the push-forward of the noise, and the noise is invariant under g.', uses: ['transport'] },
      ],
    },
    after: ['With the data at all orientations, the two molecules on the left agree to about 10⁻¹⁶ Å at every rotation and every t, which is rounding error. The data are a mixture over 360 orientations 1° apart, but with a noise of 0.08 Å per atom neighbouring orientations overlap so much that the mixture is invariant under every rotation to far below machine precision, and so is its field. The 300 generated molecules point in all directions, with a spread of 103° against 104° for the uniform distribution.', 'Trained on one orientation, the field is not equivariant: at θ = 115° the two molecules end 1.32 Å apart, and the generated molecules point within a spread of 10° of the one orientation in the data. A real model reaches equivariance in one of two ways, by an equivariant network such as the graph network of EDM, or approximately, by training a general network on randomly rotated copies of the data, which is what the left plot does when the data contain all orientations. AlphaFold 3 and SimpleFold take the second route.'],
    check: {
      q: 'The noise N(0, I) is invariant under rotations. Why is that not enough for the generated molecules to be invariant when the model is trained on one orientation?',
      opts: [
        ['The noise only fixes the distribution at t = 0. The field carries it to t = 1, and a field that is not equivariant moves rotated noises to unrotated places, here toward the single orientation it has seen.', true],
        ['Because the noise is invariant only in distribution, not for each sample.', false],
        ['It is enough; the histogram is narrow only because 300 samples are too few.', false],
      ],
      fb: { true: 'Right. The proposition needs both: an invariant source and an equivariant field. With an invariant data distribution, the exact field is automatically equivariant, which is the converse direction of the proposition.', false: 'The invariance of the marginals is carried from t = 0 to t = 1 by the field. Look at what the derivation needs from the field.' },
    },
  },
  {
    id: 'frames', kicker: 'Rigid parts, curved space', title: 'Protein backbones on SE(3)', scene: 'protein',
    prose: [
      'The N, Cα and C atoms of a residue form a nearly rigid triangle, so a backbone of N residues can be generated as N rigid frames: a position \\(\\x_i\\) of the Cα atom and a rotation \\(R_i\\). FrameDiff and FrameFlow generate on this space, \\(\\mathrm{SE}(3)^N\\). The positions move on straight lines, as in every flow in this survey. The rotations cannot: the straight line between two rotation matrices leaves the rotations, and the path follows the geodesic of SO(3) instead.',
      'The left plot moves a 12-residue α-helix from noise to its target along these paths, with uniformly random starting rotations. The toggle moves the three atoms of every residue on straight lines instead. The right plot shows the rotation of one residue, the geodesic as solid axes and the average of the two matrices as dashed ones. Move t.',
    ],
    tex: `${R}^${t} = ${R}^0 \\exp(${t}\\,${Om}),\\quad ${Om} = \\log\\big((${R}^0)^{\\top} ${R}^1\\big),\\qquad \\dot ${R}^${t} = \\frac{${R}^${t} \\log\\big((${R}^${t})^{\\top} ${R}^1\\big)}{1 - ${t}},\\qquad \\dot{${xi}}^${t} = ${xi}^1 - ${xi}^0`,
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
    after: ['On the frame path the bond lengths inside each residue are exact at every t, and the Cα RMSD to the target falls linearly from 8.05 Å to 0. When the atoms move on straight lines, the endpoints are the same, but in between the bonds stretch and shrink: the largest bond-length error is 44% at t = 0.25 and 72% at t = 0.5. The model would have to learn the rigid geometry that the frame representation gets for free.', 'For the residue on the right the angle between its noise and target frames is θ = 120°, and the geodesic turns it at a constant rate, so the angle left at time t is (1 − t)θ. The average of the two matrices at t = 0.5 is not a rotation: its determinant is 0.25 and ‖MᵀM − I‖ is 1.06. The flow-matching loss on rotations compares the predicted and the true Ω in the Frobenius norm, which for this residue is √2 θ = 2.96.'],
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
    id: 'crps', kicker: 'Forecasts are distributions', title: 'Weather and the CRPS', scene: 'crps',
    prose: [
      'GenCast generates an ensemble of weather forecasts with a conditional diffusion model, and an ensemble is judged by the continuous ranked probability score. The CRPS rewards members that are close to the observation y and subtracts half the mean distance between members, which rewards spread. It is a proper score: its expectation over observations from the true distribution G is smallest when the forecast is G itself.',
      'The forecast here is the true distribution scaled about its mean by a spread factor κ, so κ < 1 is overconfident and κ = 1 is right. In practice the CRPS is estimated from M members. The plain estimator replaces both expectations by averages over the members; the fair estimator excludes the pairs of a member with itself. Move κ and M, and switch between the two.',
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
    after: ['The expected CRPS is 1.009 for the deterministic forecast at κ = 0, 0.703 at κ = 0.7 and 0.678 at κ = 1, its minimum, and it rises again to 0.726 at κ = 1.5. The squared error of the forecast mean is the same for every κ, because every κ has the same mean, so it cannot tell a calibrated ensemble from an overconfident one. The CRPS can.', 'The plain estimator is minimized at κ = 0.49 for M = 2, 0.76 for M = 5, 0.87 for M = 10 and 0.97 for M = 50. Used as a training loss, as in FGN and Swift, it would train a small ensemble to be overconfident; the fair estimator is minimized at κ = 1 for every M. This is the general lesson of the chapter in miniature: in each application, the quantity that matters is a property of the whole distribution, and the training objective has to measure that property and not a proxy that a single sample or a mean can satisfy.'],
    check: {
      q: 'Why does the plain estimator prefer an overconfident ensemble when M is small?',
      opts: [
        ['Its spread term counts the M zero distances of each member to itself, so it rewards spread by only 1 − 1/M of what the CRPS does, and a narrower ensemble is penalized less than it should be.', true],
        ['With few members the ensemble mean is noisy, and a narrow ensemble reduces that noise.', false],
        ['Because the observation is a single number and an ensemble of M numbers cannot be compared with it fairly.', false],
      ],
      fb: { true: 'Right. The accuracy term is unbiased in both estimators; only the spread term differs. With M = 2 the plain estimator halves the reward for spread and its minimum moves to κ = 0.49.', false: 'Look at the second line of the derivation: which terms of the double sum are zero whatever the members are?' },
    },
    closing: `<h3>Where this leads</h3>
      <p>Each step isolated one design choice of an application. The table collects the measured numbers at the settings named in the text.</p>
      <table class="tbl"><thead><tr><th>Step</th><th>Comparison</th><th>Result</th></tr></thead><tbody>
        <tr><td>1. Video</td><td>joint, own noise, shared noise</td><td>clips with the right speed: 100%, 20%, 0%</td></tr>
        <tr><td>2. World model</td><td>model A against B, ε = 0.05, 64 frames</td><td>error 0.768 against 0.050</td></tr>
        <tr><td>3. Policy</td><td>regression against flow, H = 8</td><td>passes 16% against 99.8%</td></tr>
        <tr><td>4. Molecule</td><td>all orientations against one</td><td>gap 10⁻¹⁶ Å against 1.32 Å</td></tr>
        <tr><td>5. Protein</td><td>frames against straight atoms, t = 0.5</td><td>bond error 0 against 72%</td></tr>
        <tr><td>6. Weather</td><td>fair against plain estimator, M = 5</td><td>best κ 1 against 0.76</td></tr>
      </tbody></table>
      <p>Chapter 7 also covers applications that this page leaves out. Crystals live on a torus, so the noise is a wrapped Gaussian and paths take the shortest way modulo one. Medical imaging is mostly an inverse problem, solved with the posterior samplers of <a href="../inference-time/">Inference-time scaling</a> (§6). Audio and video are generated jointly by models that share attention across both streams. The fields behind every step here are those of <a href="../flow-matching/">Flow matching</a> (§2.4) and <a href="../couplings/">Couplings, transport and control</a> (§2.6); a guided or reward-tuned version of each model is the subject of <a href="../guidance/">Conditioning and guidance</a> (§2.11) and <a href="../post-training/">Reward fine-tuning</a> (§4), and the few-step generators that make real-time world models and policies possible are those of <a href="../distillation/">Distillation</a> (§5). Animations <a href="../../animations.html#a12">A12, self forcing</a>, <a href="../../animations.html#a15">A15, an action-conditioned world model</a>, and <a href="../../animations.html#a16">A16, protein frames on SE(3)</a>, show steps 2 and 5 at a larger scale.</p>`,
  },
];
