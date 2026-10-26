// Content of the §3 tutorial: one entry per step. Formulas follow paper/sections/03_finetuning.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), b = S('b'), W = S('W'), W0 = S('W0'), u = S('u'), G = S('G'), dW = S('dW'), r = S('r'), al = S('alpha'), mu = S('mu'), V = S('V');
const cs = S('cs'), cc = S('cc'), kap = S('kap'), Wid = S('Wid'), lam = S('lam'), dS = S('dS'), dT = S('dT'), WJ = S('WJ');

export const INTRO = `
  <div class="kicker">Survey §3 · Interactive tutorial</div>
  <h1>Fine-tuning and adaptation</h1>
  <p>A pretrained diffusion or flow model is adapted to a new task by changing its weights a little: a new subject, a new style, a new condition. This tutorial asks three questions that come up with every such change. How few numbers does the change need? Why does teaching the model a new subject damage what it already knew, and how does the prior-preservation loss of DreamBooth prevent it? When can two adapters that were trained separately simply be added?</p>
  <p>To answer them exactly, the model on this page is linear in its weights. At each of the 16 Euler steps of the sampler the velocity is \\(\\bm v_k(\\x) = \\W_k\\,\\bm b(\\x)\\), where \\(\\bm b\\) are 256 fixed Gaussian bumps on a grid and a constant, so the weights form one matrix of 32 × 257 numbers. Every model is the least-squares fit of these weights to the exact velocity of a Gaussian mixture in the plane, so there is no training noise, and the fine-tuning methods become linear algebra with closed-form answers. The pretrained model fits a ring of six modes. The convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\).</p>
  <ol class="toc">
    <li><a href="#lora">How many directions does an update need?</a></li>
    <li><a href="#dreambooth">A new subject without forgetting the class</a></li>
    <li><a href="#merge">When adapters add up</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'lora', kicker: 'Low-rank adaptation', title: 'How many directions does an update need?', scene: 'lora',
    prose: [
      'The target of this step replaces one mode of the ring by a new mode at its centre. Fitting all weights to it gives the full update \\(\\Delta\\W\\), a matrix of 8224 numbers. LoRA instead freezes the pretrained weights and trains a product of two thin matrices, so the update has rank at most r and costs r(M + D) = 289 r numbers. The question is how much of the full update survives this restriction.',
      'What matters is not the size of the weights but the change of the field they produce. Measured on the points where the model is used, the best rank-r update keeps the r directions of the output that carry the most change, the top eigenvectors of \\(\\Delta\\W\\,\\bm G\\,\\Delta\\W^\\top\\). Move r and watch the samples on the left and the captured share on the right. Then move α, the scale at which the adapter is applied.',
    ],
    tex: `${W} = ${W0} + ${al}\\,\\Delta${W}_{${r}}, \\qquad \\Delta${W}_{${r}} = ${V}${V}^\\top ${dW} \\quad\\text{minimizes}\\quad \\E\\big\\|(${dW} - \\Delta)\\,${b}(${x})\\big\\|^2 \\ \\text{over rank} \\le ${r}`,
    deriv: {
      have: ['The full update \\(\\Delta\\W\\) and the feature second moment \\(\\bm G = \\E[\\bm b\\,\\bm b^\\top]\\) on the design set.', 'A candidate update Δ of rank at most r.'],
      goal: '\\min_{\\operatorname{rank}\\Delta \\le r} \\E\\big\\|(\\Delta\\W - \\Delta)\\,\\bm b\\big\\|^2 = \\sum_{j > r} \\mu_j',
      plan: 'Write the error of the field as a Frobenius norm and apply the best low-rank approximation of a matrix.',
      title: 'The best update of rank r', lines: [
        { tex: `\\E\\big\\|(\\Delta\\W - \\Delta)\\,\\bm b\\big\\|^2 = ${CHG('\\operatorname{tr}\\big((\\Delta\\W - \\Delta)\\,\\bm G\\,(\\Delta\\W - \\Delta)^\\top\\big)')}`, why: 'The squared norm of a vector is the trace of its outer product, and the expectation moves inside the trace onto \\(\\bm b\\,\\bm b^\\top\\).', uses: ['expectation'] },
        { tex: `= ${CHG('\\big\\|\\Delta\\W\\,\\bm G^{1/2} - \\Delta\\,\\bm G^{1/2}\\big\\|_F^2')}`, why: 'Split \\(\\bm G = \\bm G^{1/2}\\bm G^{1/2}\\). The error of the field is the Frobenius error of the matrix \\(\\Delta\\W\\,\\bm G^{1/2}\\).' },
        { tex: `\\ge ${CHG('\\sum_{j > r} \\mu_j')}`, why: '\\(\\Delta\\,\\bm G^{1/2}\\) has rank at most r. By the Eckart–Young theorem no such matrix is closer than the truncated singular value decomposition, whose error is the sum of the discarded squared singular values. These are the eigenvalues μ_j of \\(\\Delta\\W\\,\\bm G\\,\\Delta\\W^\\top\\).' },
        { tex: `\\text{with equality at } ${CHG('\\Delta = \\bm V_r \\bm V_r^\\top \\Delta\\W')}`, why: 'The truncated decomposition projects the rows onto the top r left singular vectors \\(\\bm V_r\\), which are the top eigenvectors of \\(\\Delta\\W\\,\\bm G\\,\\Delta\\W^\\top\\). Multiplying by \\(\\bm G^{-1/2}\\) on the right gives the update.' },
      ],
    },
    after: [
      'The first direction carries 50% of the change of the field and the second 46%, so two directions hold 96% of it, and four hold 99%. Measured in parameter space instead, by the plain sum of squared weights, the same update looks less concentrated: 77% in two directions and 92% in three. A low-rank adapter works when the change is concentrated where the model is used, which is a weaker requirement than a concentrated change of the weights.',
      'The samples tell the same story with a twist. At r = 1 the replaced mode is already gone, but only 3% of the samples reach the centre, and the sliced Wasserstein distance to the target is 0.188. At r = 2 the centre mode holds 27% and the distance is 0.171; at r = 4 the distance is 0.079 and at full rank 0.070, the floor set by the fit and the 16-step sampler. The scale α interpolates: at r = 1 the replaced mode keeps 17% of the mass at α = 0, 5% at α = 0.5 and none at α = 1. Removing a mode is a large, simple change of the field, while placing a new one needs more directions.',
    ],
    check: {
      q: 'The top four directions capture 99% of the change of the field. How many numbers does a rank-4 adapter train?',
      answer: 1156, tol: 0.5,
      fb: { true: 'Right: 4 × (32 + 257) = 1156, against 8224 for the full update. In a network the saving is far larger, since D and M are in the thousands and r stays small.', false: 'A rank-r update is the product of an M × r and an r × D matrix, so it has r(M + D) entries. Here M = 32 and D = 257.' },
    },
  },
  {
    id: 'dreambooth', kicker: 'Subject-driven generation', title: 'A new subject without forgetting the class', scene: 'dreambooth',
    prose: [
      'DreamBooth teaches a model a specific subject from a handful of images, bound to a rare identifier token in a prompt like “a photo of sks dog”. The subject prompt shares almost everything with the class prompt “a photo of a dog”, so the gradient that pulls the subject prompt to the subject also pulls the class prompt there. After fine-tuning, every dog looks like the subject. This is called language drift, and the prior-preservation term fights it by asking the class prompt to keep producing what the pretrained model produced.',
      'In the toy, the class prompt sees the features \\([\\bm b, 0]\\) and the subject prompt \\([\\bm b, \\kappa\\,\\bm b]\\), where κ is how strongly the identifier enters. The subject is a single blob at the centre of the ring. Training runs k full-batch gradient steps from the pretrained weights. Start at λ = 0 and watch the class prompt on the left, then raise λ.',
    ],
    tex: `\\E\\big\\|\\hat${u}(${x}_t, ${cs}) - ${u}_{\\mathrm{subj}}\\big\\|^2 + ${lam}\\,\\E\\big\\|\\hat${u}(${x}_t, ${cc}) - \\hat${u}_{${W0}}(${x}_t, ${cc})\\big\\|^2, \\qquad \\hat${u}(\\cdot, ${cs}) = (\\W_{\\mathrm s} + ${kap}\\,${Wid})\\,${b}`,
    deriv: {
      have: ['Class weights \\(\\W_{\\mathrm s}\\) shared by both prompts, started at \\(\\W_0\\), and identifier weights \\(\\W_{\\mathrm{id}}\\), started at 0.', 'The loss at λ = 0, \\(L = \\E\\|(\\W_{\\mathrm s} + \\kappa\\,\\W_{\\mathrm{id}})\\,\\bm b - \\vel_{\\mathrm{subj}}\\|^2\\), minimized by gradient descent.'],
      goal: '\\W_{\\mathrm s} - \\W_0 = \\frac{1}{1 + \\kappa^2}\\,\\big(\\W_{\\mathrm s} + \\kappa\\,\\W_{\\mathrm{id}} - \\W_0\\big)',
      plan: 'Show that the two gradients are always proportional, so the two blocks move together, and compare the change of the class prompt with that of the subject prompt.',
      title: 'Without prior preservation, the class takes its share', lines: [
        { tex: `\\nabla_{\\W_{\\mathrm s}} L = ${CHG('2\\,\\E\\big[\\bm e\\,\\bm b^\\top\\big]')}, \\qquad \\nabla_{\\W_{\\mathrm{id}}} L = ${CHG('\\kappa')}\\cdot 2\\,\\E\\big[\\bm e\\,\\bm b^\\top\\big]`, why: 'With the residual \\(\\bm e = (\\W_{\\mathrm s} + \\kappa\\,\\W_{\\mathrm{id}})\\,\\bm b - \\vel_{\\mathrm{subj}}\\), the identifier weights enter through κ b, so their gradient carries an extra factor κ.', uses: ['grad-linear', 'chain-rule'] },
        { tex: `\\W_{\\mathrm s} - \\W_0 = ${CHG('\\bm C')}, \\qquad \\W_{\\mathrm{id}} = ${CHG('\\kappa\\,\\bm C')}`, why: 'Every step moves both blocks along the same matrix, the second κ times as far. Both start from a difference of zero, so after any number of steps they are \\(\\bm C\\) and \\(\\kappa\\,\\bm C\\) for one accumulated matrix \\(\\bm C\\).' },
        { tex: `\\W_{\\mathrm s} + \\kappa\\,\\W_{\\mathrm{id}} - \\W_0 = ${CHG('(1 + \\kappa^2)\\,\\bm C')}`, why: 'The subject prompt sees the sum, so its change is the class change plus κ² times it.' },
        { tex: `\\W_{\\mathrm s} - \\W_0 = ${CHG('\\frac{1}{1 + \\kappa^2}')}\\,\\big(\\W_{\\mathrm s} + \\kappa\\,\\W_{\\mathrm{id}} - \\W_0\\big)`, why: 'Divide. Whatever the subject prompt learns, the class prompt learns the fraction 1/(1 + κ²) of it. With a weak identifier, small κ, the class prompt learns nearly all of it.' },
      ],
    },
    after: [
      'At λ = 0, κ = 0.5 and k = 3000 the class prompt receives 80% of the update. Its samples crowd towards the centre, at a sliced Wasserstein distance of 0.59 from the ring against 0.08 for the pretrained model. With κ = 0.3 it is worse: 92% of the update, a distance of 0.65, and 82% of the class samples within 0.6 of the centre. With κ = 1 the class receives half the update, which pulls the ring in without filling the centre, at a distance of 0.35. Training longer makes the subject sharper and the drift worse: at κ = 0.5 the share of class samples near the centre grows from 2% at k = 1000 to 34% at k = 30000.',
      'Prior preservation fixes this at a small price. At κ = 0.5 and λ = 1 the class prompt stays at 0.09 from the ring, close to the 0.08 of the pretrained model, and the subject prompt reaches its target at 0.12 instead of 0.10. The term needs no data of the class: in practice it is computed on samples that the frozen model generates for the class prompt, which is what the second expectation means.',
    ],
    check: {
      q: 'Without prior preservation, what share of the subject’s update does the class prompt receive at κ = 1?',
      answer: 0.5, tol: 0.01,
      fb: { true: 'Right: 1/(1 + 1²) = 0.5. In the plot the ring is pulled halfway to the subject, which is why the class distance at κ = 1 and λ = 0 is about half of that at κ = 0.3.', false: 'Use the last line of the derivation, the fraction 1/(1 + κ²).' },
    },
  },
  {
    id: 'merge', kicker: 'Composing adapters', title: 'When adapters add up', scene: 'merge',
    prose: [
      'Adapters are often trained separately and combined afterwards: one for a subject, one for a style, added to the same base weights. The sum is cheap and needs no further training, but it is only correct if the two edits do not interact. Here the subject adapter puts a blob at the centre of the ring, and the style adapter either rotates the ring by 20° or shrinks it to 0.6 of its radius. The joint target has the edited ring with the subject at its centre, and the right plot compares the sum with it.',
      'Because every fit on this page is least squares on the same design set, the fitted weights depend linearly on the target velocity. The error of the sum is then itself a fit, namely the fit of the part of the joint target that is not the sum of the two edits. Switch between the two styles, and try the orthogonalized adapter and the joint fit.',
    ],
    tex: `${W0} + ${al}\\,${dS} + ${al}\\,${dT} - ${WJ} = -\\mathcal{F}\\big[${u}_J - ${u}_{\\mathrm{subj}} - ${u}_{\\mathrm{style}} + ${u}_0\\big] \\quad\\text{at } ${al} = 1`,
    deriv: {
      have: ['The fit \\(\\mathcal F[\\vel] = \\E[\\vel\\,\\bm b^\\top]\\,\\bm G^{-1}\\), the least-squares weights for a target field \\(\\vel\\).', 'Targets \\(\\vel_0\\) for the base, \\(\\vel_{\\mathrm{subj}}\\) and \\(\\vel_{\\mathrm{style}}\\) for each edit, and \\(\\vel_J\\) for both.'],
      goal: '\\W_0 + \\Delta\\W_{\\mathrm{subj}} + \\Delta\\W_{\\mathrm{style}} - \\W_J = -\\mathcal F[\\vel_J - \\vel_{\\mathrm{subj}} - \\vel_{\\mathrm{style}} + \\vel_0]',
      plan: 'Write each adapter as the difference of two fits and use that the fit is linear in its target.',
      title: 'The merge error is the fit of the interaction', lines: [
        { tex: `\\Delta\\W_{\\mathrm{subj}} = ${CHG('\\mathcal F[\\vel_{\\mathrm{subj}}] - \\mathcal F[\\vel_0]')}, \\qquad \\Delta\\W_{\\mathrm{style}} = ${CHG('\\mathcal F[\\vel_{\\mathrm{style}}] - \\mathcal F[\\vel_0]')}`, why: 'Each adapter is the fine-tuned fit minus the pretrained fit \\(\\W_0 = \\mathcal F[\\vel_0]\\).' },
        { tex: `\\W_0 + \\Delta\\W_{\\mathrm{subj}} + \\Delta\\W_{\\mathrm{style}} - \\W_J = ${CHG('\\mathcal F[\\vel_{\\mathrm{subj}}] + \\mathcal F[\\vel_{\\mathrm{style}}] - \\mathcal F[\\vel_0] - \\mathcal F[\\vel_J]')}`, why: 'Substitute, with \\(\\W_J = \\mathcal F[\\vel_J]\\).' },
        { tex: `= ${CHG('-\\mathcal F[\\vel_J - \\vel_{\\mathrm{subj}} - \\vel_{\\mathrm{style}} + \\vel_0]')}`, why: '\\(\\mathcal F\\) is an expectation of the target times a fixed matrix, so it is linear and the four fits combine into one.', uses: ['expectation'] },
      ],
    },
    after: [
      'For the rotation the two adapters are nearly orthogonal in function space, with a cosine of 0.11, but 55% of the joint change is not the sum of the two changes. The sum reaches a distance of 0.122 to the joint target, against 0.074 for the joint fit, and leaves 4% of its samples at a stray mode: the subject adapter removes the first mode where it was before the rotation, while the rotated ring puts it 20° further on. For the shrink the interaction is larger. The adapters have a cosine of 0.56, because both move mass towards the centre, and the sum puts 74% of all samples near the centre, at a distance of 0.347. The joint fit reaches 0.042.',
      'Two fixes are common in practice. Scaling both adapters down reduces the overlap but also the edits: at α = 0.5 the shrink merge reaches 0.173 and still puts 11% near the centre. Orthogonalizing the style adapter against the subject adapter removes the double counting along their shared direction, which improves the shrink merge from 0.347 to 0.228, but cannot create the part of the joint change that neither adapter contains. Only a fit to the joint target, or training the adapters together, recovers it.',
    ],
    check: {
      q: 'When is the sum of two independently fitted adapters exactly the joint fit?',
      opts: [
        ['When the interaction field \\(\\vel_J - \\vel_{\\mathrm{subj}} - \\vel_{\\mathrm{style}} + \\vel_0\\) has zero fit, that is, when it is orthogonal to every feature on the design set.', true],
        ['When the two adapters are orthogonal in function space.', false],
        ['When both adapters have low rank.', false],
      ],
      fb: { true: 'Right. Orthogonal adapters are neither necessary nor sufficient: the rotation adapters have a cosine of only 0.11 and still interact, because the joint target is not the sum of the two edits.', false: 'Look at the last line of the derivation. Which quantity must vanish for the merge error to be zero?' },
    },
    closing: `<h3>Where this leads</h3>
      <p>The three steps share one picture. A fine-tuning update is a change of the field, and its size, its rank and its interactions are measured on the points where the model is used. LoRA works when that change is concentrated in a few output directions, prior preservation works because it constrains the change of the class prompt on its own samples, and adapters add up only when the edits they encode do not interact.</p>
      <p>Section 3 of the survey covers methods that this toy leaves out. Textual inversion optimizes only the embedding of the identifier token and keeps all weights fixed. ControlNet adds a trainable copy of the encoder whose output enters through convolutions initialized at zero, so that training starts exactly at the pretrained model. Editing methods use two guidance scales, one for the image and one for the instruction. Fine-tuning on few images also raises the question of memorization, which <a href="../generalization/">Generalization and memorization</a> (§2.7) studies for the training set itself. After adaptation comes alignment: <a href="../post-training/">Reward fine-tuning and preference alignment</a> (§4) tilts the adapted model towards a reward. Animation <a href="../../animations.html#a6">A6, LoRA in weight space</a>, makes the edit of step 1 with a trained network instead of the linear model of this page.</p>`,
  },
];
