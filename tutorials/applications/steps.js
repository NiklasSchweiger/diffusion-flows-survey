// Content of the §7 tutorial: one entry per step. Formulas follow paper/sections/07_applications.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const t = S('t'), x = S('x'), F = S('F'), k = S('k'), eps = S('eps'), K = S('K'), e = S('e'), L = S('L');
const o = S('o'), A = S('A'), H = S('H'), tau = S('tau'), g = S('g'), Phi = S('Phi'), xi = S('xi'), R = S('R'), Om = S('Om');
const y = S('y'), Gt = S('Gt'), kap = S('kap'), M = S('M');

export const INTRO = `
  <div class="kicker">Survey §7 · Interactive tutorial</div>
  <h1>Applications</h1>
  <p>Images are the easy case for a generative model: one sample, no symmetry to respect and no consequences once it is drawn. The applications of chapter 7 add structure that the model has to get right. A video is many frames that must agree with each other. A world model and a robot policy generate step by step, and each step conditions on what the model produced before. Molecules and proteins have symmetries and rigid parts, and a weather forecast is judged as a distribution rather than as a single picture.</p>
  <p>Each step isolates one of these issues. Steps 1, 2, 4 and 5 run on real data: photographs by Eadweard Muybridge, a maze game that you play, drug molecules from PubChem and protein backbones from the Protein Data Bank. No network is trained anywhere. Every model is the exact field or conditional of its data, so every difference on the canvas comes from the design choice and none from training. Steps 1, 4, 5 and 6 are exact up to the ODE solver, step 2 is exact, and step 3 is Monte Carlo over seeded rollouts of an exact conditional. The convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\); the flow time of the robot policy is written τ.</p>
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
      'A video model has to generate frames that belong together. The data here are real: six photographic sequences by Eadweard Muybridge, of a racing, galloping, pacing, walking and jumping horse and a galloping buffalo, reduced to 128 × 80 grey pixels. A training clip is a window of F = 8 consecutive frames, which gives 62 clips built from 104 distinct frames.',
      'No network is trained. Every model is the exact denoiser of its own training set, which for a finite set is a softmax-weighted average of the training items. The joint model treats a clip as one vector of 8 × 2560 pixels and averages whole training clips. The two per-frame models only know the 104 single frames and denoise each frame of a clip on its own, with independent noise per frame or with one noise shared by all frames. Pick a model, move t and press Play to watch the generated clips.',
    ],
    tex: `\\hat${x}_1(${x}_${t}) = \\sum_n w_n\\,${x}^{(n)},\\qquad w_n \\propto \\exp\\Big(-\\frac{\\|${x}_${t} - ${t}\\,${x}^{(n)}\\|^2}{2(1 - ${t})^2}\\Big),\\qquad ${x} = (x^1, \\dots, x^{${F}})`,
    after: ['All 12 clips of the joint model are consecutive frames of one training clip, and they play as smooth motion. With independent noise, every single frame is a sharp training frame, because each frame’s denoiser is exact, but the frames of one clip come from 4.25 different sequences on average and none of the 12 clips is in order. Played back, they jump from horse to horse. With one noise for all frames, every frame of a clip starts from the same point and follows the same field, so it reaches the same training frame: all 12 clips are one still image repeated eight times.', 'Both failures have one cause. The order of the frames is a property of the joint distribution of a clip, and a model of single frames cannot recover it, whatever noise its frames share. Video models therefore denoise all frames of a clip together, with attention across time, and generate long videos chunk by chunk, each chunk conditioned on the previous one. The step also shows the price of an exact model: the right grid names the training frame behind every generated frame, and the joint model only replays its training clips. The exact denoiser of a finite set memorizes it, and why a trained network generates new clips instead is the subject of <a href="../generalization/">Why diffusion models generalize</a> (§2.7).'],
    check: {
      q: 'The per-frame model with shared noise produces clips that stand still, and the one with independent noise produces clips that jump. Which change would make a per-frame model produce clips that move?',
      opts: [
        ['None that keeps it a model of single frames. A per-frame model only knows which frames are likely, not which frame follows which, and that order is a property of the joint distribution.', true],
        ['Sharing the noise only between neighbouring frames, so the clip is smooth and can still change over time.', false],
        ['Training the per-frame model longer, so it learns the single frames more accurately.', false],
      ],
      fb: { true: 'Right. The joint model differs from the per-frame models only in its field, which looks at all frames of the clip at once. Partially shared noise interpolates between jumping and standing still, but it has no information about which frame comes next.', false: 'The per-frame denoisers here are already exact. Ask what a model of single frames would need to know to put frame 4 after frame 3.' },
    },
  },
  {
    id: 'world', kicker: 'Generating on its own output', title: 'World models and compounding error', scene: 'world',
    prose: [
      'A world model generates the next frame from the previous frames and the action of a player, one frame after the other, \\(p(\\x^{1:K}\\mid\\x^0, a^{1:K}) = \\prod_k p(\\x^k\\mid\\x^{\\lt k}, a^k)\\). Here the game is a maze, and you play it. Click the game and steer with the arrow keys, or use the buttons. The true game moves the player one cell in the direction of the key unless a wall is in the way. Both models are specified by hand and are exact: each moves a share 1 − ε of the probability correctly and spills the rest one cell too far or leaves it behind.',
      'After the first frame, neither model sees the true game again; each conditions on the frame it generated last. Model B differs from model A only off the data: before it steps, it snaps its context to the most likely cell, which leaves every real frame unchanged. Both therefore have the same one-step error ε on real contexts, the quantity that teacher-forced training minimizes. Play for a while and compare the three screens.',
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
    after: ['In the demo run of 40 inputs at ε = 0.05, the error of model A is 0.085 after 5 frames, 0.288 after 10, 0.520 after 30 and 0.668 after 40. On its screen the player dissolves into faint copies spread along the corridors. For the first four frames its error grows by almost ε per frame, as fast as the linear bound kε allows. A wall sometimes takes part of it back, when the spilled probability has nowhere to go but the true cell, but most of it is never corrected, so A behaves like L = 1. The bound itself is vacuous from frame 20 on. The error of model B never exceeds ε at any frame and for every ε on the slider: it reads its own blurred frame as the nearest valid frame, so its errors do not accumulate (L = 0 near the data).', 'Both models have the same teacher-forced loss, and only a rollout on generated context tells them apart. This is why world models are trained with corrupted or self-generated contexts: GameNGen adds noise to the context frames, Diffusion Forcing trains with independent noise levels per frame and Self Forcing trains on the model’s own rollouts. Each of these teaches the model what to do with a context that is slightly off.'],
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
      'A molecule is the same molecule after a rotation, so a generative model of molecules should give \\(\\x\\) and \\(g\\x\\) the same probability. The proposition of §7.6 says how a flow achieves this: if the noise distribution is invariant under the rotations and the velocity is equivariant, every marginal of the flow is invariant. The left plot tests the equivariance directly. It flows a rotated noise (solid) and rotates the flow of the original noise (dashed rings); if the field is equivariant, the two coincide at every t.',
      'The molecules are real 3D conformers from PubChem, hydrogens included: caffeine, aspirin, ibuprofen, paracetamol and nicotine. The data of one molecule are its conformer at a uniformly random rotation with a small noise of 0.1 Å per coordinate, and the field is the exact velocity of that distribution. Translations are removed by centring the molecule, because they have no invariant probability distribution. Pick a molecule, move θ and t, and drag to turn the view. Then train on a single orientation and compare.',
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
    after: ['With the data at all orientations, the two molecules on the left agree to within 10⁻¹⁴ Å for all five molecules, at every rotation and every t, which is rounding error. The field is exactly equivariant: given \\(\\x_t\\) it averages over the posterior of the rotation, which it evaluates on rotations around its mode, and rotating \\(\\x_t\\) rotates the mode with it. The 32 generated molecules on the right point in all directions. Their mean rotation from the conformer is between 122° and 137° for the five molecules, against 126.5° for uniform rotations.', 'Trained on one orientation, the field is not equivariant: at θ = 115° the two caffeine molecules end 6.4 Å apart, and every generated molecule lies within 2° of the one orientation in the data. A real model reaches equivariance in one of two ways, by an equivariant network such as the graph network of EDM, or approximately, by training a general network on randomly rotated copies of the data, which is what the data at all orientations amount to here. AlphaFold 3 and SimpleFold take the second route.'],
    check: {
      q: 'The noise N(0, I) is invariant under rotations. Why is that not enough for the generated molecules to be invariant when the model is trained on one orientation?',
      opts: [
        ['The noise only fixes the distribution at t = 0. The field carries it to t = 1, and a field that is not equivariant moves rotated noises to unrotated places, here toward the single orientation it has seen.', true],
        ['Because the noise is invariant only in distribution, not for each sample.', false],
        ['It is enough; the samples look aligned only because 32 samples are too few.', false],
      ],
      fb: { true: 'Right. The proposition needs both: an invariant source and an equivariant field. With an invariant data distribution, the exact field is automatically equivariant, which is the converse direction of the proposition.', false: 'The invariance of the marginals is carried from t = 0 to t = 1 by the field. Look at what the derivation needs from the field.' },
    },
  },
  {
    id: 'frames', kicker: 'Rigid parts, curved space', title: 'Protein backbones on SE(3)', scene: 'protein',
    prose: [
      'The N, Cα and C atoms of a residue form a nearly rigid triangle, so a backbone of N residues can be generated as N rigid frames: a position \\(\\x_i\\) of the Cα atom and a rotation \\(R_i\\). FrameDiff and FrameFlow generate on this space, \\(\\mathrm{SE}(3)^N\\). The positions move on straight lines, as in every flow in this survey. The rotations cannot: the straight line between two rotation matrices leaves the rotations, and the path follows the geodesic of SO(3) instead.',
      'The targets are real backbones from the Protein Data Bank: ubiquitin (1UBQ, 76 residues), the engrailed homeodomain (1ENH, 54 residues) and the Trp-cage miniprotein (1L2Y, 20 residues). Each residue’s frame comes from its own N, Cα and C atoms. The path starts from Cα positions drawn from a Gaussian of the protein’s size and from uniformly random rotations. The toggle moves the three atoms of every residue on straight lines instead. The right plot shows the rotation of one residue, the geodesic as solid axes and the average of the two matrices as dashed ones. Move t or press Play, and drag the protein to turn the view.',
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
    after: ['On the frame path the bond lengths inside every residue are exact at every t, to 10⁻¹⁵, and the Cα RMSD to the target falls linearly, for ubiquitin from 14.9 Å to 0. When the atoms move on straight lines, the endpoints are the same, but in between the bonds stretch and shrink: the largest bond-length error is about 50% at t = 0.25 for all three proteins and, at t = 0.5, 92% for ubiquitin, 98% for the homeodomain and 88% for the Trp-cage. The model would have to learn the rigid geometry that the frame representation gets for free. On both paths the helices and strands appear only close to t = 1; they are a property of the target, not of the path.', 'For the residue on the right, residue 57 of ubiquitin, the angle between its noise and target frames is θ = 120°, and the geodesic turns it at a constant rate, so the angle left at time t is (1 − t)θ. The average of the two matrices at t = 0.5 is not a rotation: its determinant is 0.25 and ‖MᵀM − I‖ is 1.06. The flow-matching loss on rotations compares the predicted and the true Ω in the Frobenius norm, which for this residue is √2 θ = 2.96.'],
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
        <tr><td>1. Video</td><td>joint, own noise, shared noise</td><td>clips in the order of a training clip: 12, 0, 0 of 12; with shared noise all 12 stand still</td></tr>
        <tr><td>2. World model</td><td>model A against B, ε = 0.05, demo run of 40 frames</td><td>error 0.668 against at most 0.050</td></tr>
        <tr><td>3. Policy</td><td>regression against flow, H = 8</td><td>passes 16% against 99.8%</td></tr>
        <tr><td>4. Molecule</td><td>all orientations against one, caffeine, θ = 115°</td><td>gap below 10⁻¹⁴ Å against 6.4 Å</td></tr>
        <tr><td>5. Protein</td><td>frames against straight atoms, ubiquitin, t = 0.5</td><td>bond error 0 against 92%</td></tr>
        <tr><td>6. Weather</td><td>fair against plain estimator, M = 5</td><td>best κ 1 against 0.76</td></tr>
      </tbody></table>
      <p>Chapter 7 also covers applications that this page leaves out. Crystals live on a torus, so the noise is a wrapped Gaussian and paths take the shortest way modulo one. Medical imaging is mostly an inverse problem, solved with the posterior samplers of <a href="../inference-time/">Inference-time scaling</a> (§6). Audio and video are generated jointly by models that share attention across both streams. The fields behind every step here are those of <a href="../flow-matching/">Flow matching</a> (§2.4) and <a href="../couplings/">Couplings, transport and control</a> (§2.6); a guided or reward-tuned version of each model is the subject of <a href="../guidance/">Conditioning and guidance</a> (§2.11) and <a href="../post-training/">Reward fine-tuning</a> (§4), and the few-step generators that make real-time world models and policies possible are those of <a href="../distillation/">Distillation</a> (§5). Animations <a href="../../animations.html#a12">A12, self forcing</a>, <a href="../../animations.html#a15">A15, an action-conditioned world model</a>, and <a href="../../animations.html#a16">A16, protein frames on SE(3)</a>, show steps 2 and 5 at a larger scale.</p>`,
  },
];
