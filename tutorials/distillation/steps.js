// Content of the §5 tutorial: one entry per step. Formulas follow paper/sections/05_distillation.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), t = S('t'), s = S('s'), u = S('u'), X = S('X'), F = S('F'), pd = S('pd'), K = S('K'), h = S('h');
const f = S('f'), v = S('v'), ub = S('ub'), gth = S('gth'), sr = S('sr'), sf = S('sf'), lam = S('lam');

export const INTRO = `
  <div class="kicker">Survey §5 · Interactive tutorial</div>
  <h1>Distillation and few-step generation</h1>
  <p>A diffusion or flow model samples by solving an ODE, and a good solution takes tens of network evaluations. Distillation trains a student that covers the same path in a few steps or in one. This tutorial first defines the object that every student approximates, the flow map that jumps along the ODE from one time to another, and measures what the ODE costs with a plain Euler solver. It then follows the main families of students: progressive distillation, which halves the number of steps in rounds; consistency models, which map every point of a path to its end; reflow, which straightens the paths themselves; average-velocity models, which learn the jump directly; and distribution matching, which trains a one-step generator with two scores instead of paired targets.</p>
  <p>The teacher is the exact probability-flow ODE of the three-mode mixture of §2.1. In 1D this ODE keeps the order of points, so its flow map has a closed form through the CDF of the noised mixture, and the teacher has no error of its own. Students are piecewise-linear maps fitted by least squares in the page, and generated sets are 2000 stratified quantiles of the noise, so the Wasserstein-1 distance W1 to the data has no Monte Carlo noise. Step 5 moves to the 2D mixture of §2.2. The convention is that of the survey, with \\(t = 0\\) noise, \\(t = 1\\) data and \\(\\x_t = t\\,\\x_1 + (1 - t)\\,\\x_0\\).</p>
  <ol class="toc">
    <li><a href="#flowmap">The flow map</a></li>
    <li><a href="#euler">The cost of the ODE</a></li>
    <li><a href="#progressive">Progressive distillation</a></li>
    <li><a href="#consistency">Consistency models</a></li>
    <li><a href="#reflow">Reflow</a></li>
    <li><a href="#meanflow">Average velocities</a></li>
    <li><a href="#dmd">Matching distributions</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'flowmap', kicker: 'The object', title: 'The flow map', scene: 'flowmap',
    prose: [
      'The probability-flow ODE moves every point along a fixed path from noise to data. The flow map collects all of these paths into one function of two times: it takes a point at time s and returns where its path is at time t. Sampling is the flow map from 0 to 1 applied to noise, and every distillation method in this tutorial tries to evaluate that map, or a part of it, with few network calls.',
      'In general the flow map is only available by solving the ODE. In 1D the paths never cross, so a point keeps its rank: if a fraction q of the mass lies below it at time s, the same fraction lies below it at time t. The map is therefore the quantile function of \\(p_t\\) applied to the CDF of \\(p_s\\). Move s and t to see one jump on the left and the whole map on the right.',
    ],
    tex: `${X}(${x}) = ${x} + \\int_{${s}}^{${t}} ${u.replace('_t', '_\\tau')}(\\x_\\tau)\\,\\dd\\tau = ${F}^{-1}\\big(F_${s}(${x})\\big)`,
    deriv: {
      have: ['The ODE \\(\\dot\\x_\\tau = \\vel_\\tau(\\x_\\tau)\\) carries \\(p_s\\) to \\(p_t\\) (the continuity equation of §2.4).', 'In 1D, two solutions of the ODE cannot cross, so the map \\(X_{s\\to t}\\) is increasing.'],
      goal: 'X_{s\\to t} = F_t^{-1}\\circ F_s',
      plan: 'Compute the CDF of the pushed-forward points in two ways.',
      title: 'The 1D flow map from CDFs', lines: [
        { tex: `F_t\\big(X_{s\\to t}(\\x)\\big) = P\\big(\\x_t \\le X_{s\\to t}(\\x)\\big)`, why: 'The definition of the CDF of \\(p_t\\), the law of \\(\\x_t\\).', uses: ['expectation'] },
        { tex: `= P\\big(${CHG('X_{s\\to t}(\\x_s)')} \\le X_{s\\to t}(\\x)\\big)`, why: 'The ODE carries \\(p_s\\) to \\(p_t\\), so \\(\\x_t\\) has the law of \\(X_{s\\to t}(\\x_s)\\).', uses: ['transport', 'pf-ode'] },
        { tex: `= P\\big(${CHG('\\x_s \\le \\x')}\\big) = ${CHG('F_s(\\x)')}`, why: 'An increasing map keeps the order of points, so the event is the same.', uses: ['substitution'] },
        { tex: `\\Rightarrow\\ X_{s\\to t}(\\x) = ${CHG('F_t^{-1}\\big(F_s(\\x)\\big)')}`, why: 'Apply the inverse of \\(F_t\\), which exists because \\(p_t\\) is positive everywhere.', uses: [] },
      ],
    },
    after: ['At s = 0.2 the marked start 0.212 jumps to 0.244 at t = 0.7. Continuing from there to t = 1 gives 0.336, the same value as the single jump from 0.2 to 1: jumps compose, \\(X_{t\\to 1}\\circ X_{s\\to t} = X_{s\\to 1}\\). The derivative of the jump in t at t = s is −0.286, which is the velocity \\(\\vel_{0.2}(0.212)\\). These two properties, composition and the velocity as the infinitesimal jump, are what the students of steps 3, 4 and 6 are trained on.'],
    check: {
      q: 'What is \\(X_{s\\to s}(\\x)\\), and what is \\(\\partial_t X_{s\\to t}(\\x)\\) at \\(t = s\\)?',
      opts: [
        ['\\(X_{s\\to s}(\\x) = \\x\\), and the derivative is the velocity \\(\\vel_s(\\x)\\).', true],
        ['\\(X_{s\\to s}(\\x) = \\x\\), and the derivative is zero because the jump has length zero.', false],
        ['\\(X_{s\\to s}(\\x) = F_s(\\x)\\), and the derivative is the score \\(\\nabla\\log p_s(\\x)\\).', false],
      ],
      fb: { true: 'Right. A jump of length zero leaves the point where it is, and the first-order change of the jump is the velocity, by the integral form of the map. The readout shows both numbers.', false: 'Set t = s in the integral form of the map, and then differentiate the integral with respect to its upper limit.' },
    },
  },
  {
    id: 'euler', kicker: 'The baseline', title: 'The cost of the ODE', scene: 'euler',
    prose: [
      'Without a student, the flow map is approximated by a numerical solver. The simplest is the Euler method: move along the velocity at the current point for a step of size h = 1/K. Each step follows the tangent rather than the curved path and leaves a small local error, shown in red. The local error of one step is of order \\(h^2\\), and K steps add up K of them, so the error at the end is of order h.',
      'Here the field is exact, so every error on the plot is the error of the solver. The right plot gives W1 to the data for K from 1 to 128 against the exact flow map, which reaches the floor of 2000 stratified samples with a single jump. Move K and compare the paths with the exact ones.',
    ],
    tex: `\\x_{k+1} = \\x_k + ${h}\\,\\vel_{t_k}(\\x_k), \\qquad t_k = k${h},\\quad ${h} = 1/${K}`,
    after: ['With K = 1, 2, 4, 8, 16, 32, 64 and 128 steps the W1 to the data is 1.310, 0.628, 0.320, 0.165, 0.084, 0.042, 0.021 and 0.011: it halves each time K doubles, as a first-order method should. A single exact jump gives 0.0009. Reaching W1 = 0.02 takes 64 network evaluations with Euler, which is the cost that the students of the next steps try to remove. Higher-order solvers do better (§2.4 and §6), but they still need several evaluations.'],
    check: {
      q: 'Why does the error at the end fall like 1/K rather than like 1/K²?',
      opts: [
        ['Each step has a local error of order h², and K = 1/h such errors accumulate along the path.', true],
        ['The velocity is estimated from samples, and its error falls like 1/K.', false],
        ['The stratified samples have an error of order 1/K.', false],
      ],
      fb: { true: 'Right. The local errors add up (they are also carried along by the flow, which changes the constant but not the order), so the global error is K · O(h²) = O(h).', false: 'The field here is exact and the sample floor is 0.0009. Look at the red bars: how large is each one, and how many are there?' },
    },
  },
  {
    id: 'progressive', kicker: 'Halving the steps', title: 'Progressive distillation', scene: 'progressive',
    prose: [
      'Progressive distillation trains in rounds. In each round a student takes one step of size 2h where the previous model took two steps of size h, so the number of steps halves. The target of each student step is computed by running the previous model, so no data are needed. After six rounds a 64-step teacher becomes a one-step model.',
      'The students here are piecewise-linear maps with M knots per step, fitted by least squares to two steps of the previous model at 2000 points of the exact marginal. The left plot shows the error of the whole model, from noise to data, against the exact flow map. Press Next round to distil, and change M to change the capacity of the student.',
    ],
    tex: `\\text{student}_{2${h}}(\\x_{t}) \\approx \\text{model}_{${h}}\\big(\\text{model}_{${h}}(\\x_t)\\big), \\qquad \\text{fitted on } \\x_t \\sim p_t,\\ \\text{with } M \\text{ knots}`,
    after: ['The 64-step Euler teacher has W1 = 0.0213. With M = 16 the students reach 0.026 or 0.027 at every number of steps from 32 down to 1, against 1.310 for one Euler step. With M = 32 they stay between 0.023 and 0.024, with M = 8 between 0.058 and 0.063, and with M = 4 at 0.123. More capacity brings the students to the teacher but not past it: the error of the 64-step teacher is inherited by every round.'],
    check: {
      q: 'What does the one-step student converge to as M grows without bound?',
      opts: [
        ['The map of 64 Euler steps of the teacher, with W1 about 0.021.', true],
        ['The exact flow map \\(X_{0\\to 1}\\), with W1 at the sample floor.', false],
        ['The one-step Euler map, with W1 1.31.', false],
      ],
      fb: { true: 'Right. Every target is computed by the previous model, and the first previous model is the 64-step Euler teacher. A perfect student copies that teacher, including its discretization error.', false: 'Follow the targets back through the rounds. Which model produced the first targets, and does any round look at the exact map?' },
    },
  },
  {
    id: 'consistency', kicker: 'Jumping to the end', title: 'Consistency models', scene: 'consistency',
    prose: [
      'A consistency model learns the jump from any time straight to the end, \\(f(\\x_t, t) \\approx X_{t\\to 1}(\\x_t)\\), with the boundary condition that a point at t = 1 stays where it is. One evaluation at t = 0 gives a sample. For a second step, the sample is noised back to an intermediate time and the model jumps again, which corrects part of the first error.',
      'The model is trained on a time grid with gap Δ = 1/N, so that the outputs at two neighbouring points of one path agree. Consistency distillation (CD) finds the neighbouring point with one Euler step of the teacher. Consistency training (CT) needs no teacher: it uses the same pair of noise and data at both times, as in flow matching. Switch between the two and change N.',
    ],
    tex: `${f}(\\x_{t_j}, t_j) = ${f}(\\x_{t_{j+1}}, t_{j+1}) \\ \\text{along one path}, \\qquad ${f}(\\x, 1) = \\x`,
    deriv: {
      have: ['A function f that agrees at all points of each ODE path.', 'The boundary condition \\(f(\\x, 1) = \\x\\).'],
      goal: 'f(\\x_t, t) = X_{t\\to 1}(\\x_t)',
      plan: 'Walk along the path from t to the end and use the boundary condition there.',
      title: 'The consistency condition fixes f', lines: [
        { tex: `f(\\x_t, t) = f\\big(${CHG('X_{t\\to t\\prime}(\\x_t), t\\prime')}\\big) \\quad\\text{for all } t\\prime \\ge t`, why: '\\(X_{t\\to t\\prime}(\\x_t)\\) is the point of the same path at time t′, where f takes the same value.', uses: ['pf-ode'] },
        { tex: `= f\\big(X_{t\\to ${CHG('1')}}(\\x_t), ${CHG('1')}\\big)`, why: 'Take t′ = 1, the end of the path.', uses: [] },
        { tex: `= ${CHG('X_{t\\to 1}(\\x_t)')}`, why: 'The boundary condition \\(f(\\x, 1) = \\x\\).', uses: [] },
      ],
    },
    after: ['With CD the one-step W1 is 0.628, 0.322, 0.166, 0.085, 0.045 and 0.026 for N = 2, 4, 8, 16, 32 and 64. These are almost exactly the errors of N Euler steps, because the targets chain N teacher steps together. With CT the one-step W1 is 0.780, 0.451, 0.258, 0.159, 0.104 and 0.067, larger at every N. Two steps help most at small N: with CD at N = 4 they lower W1 from 0.322 to 0.261. From N = 32 on the second step adds error (0.026 to 0.035 at N = 64), because the re-noising itself draws fresh noise; the exact map with the same re-noising has 0.020.'],
    check: {
      q: 'Why is consistency training biased when Δ is not small?',
      opts: [
        ['Its target uses the straight interpolation between a fixed noise and data pair. The point at t + Δ on that line is not on the ODE path through \\(\\x_t\\); the difference vanishes only as Δ → 0.', true],
        ['It uses fewer pairs than consistency distillation.', false],
        ['It has no boundary condition at t = 1.', false],
      ],
      fb: { true: 'Right. Conditioned on \\(\\x_t\\), the average direction of the pairs is the velocity, so for small Δ the CT target is one Euler step of the exact field. For large Δ the straight lines of the pairs fan out from the curved path and the target is biased.', false: 'Compare the two targets: CD moves \\(\\x_t\\) along the ODE, and CT moves it along the straight line of one pair. When do these agree?' },
    },
  },
  {
    id: 'reflow', kicker: 'Straight paths', title: 'Reflow', scene: 'reflow',
    prose: [
      'The paths of the teacher are curved because the independent pairing of noise and data in flow matching makes them cross on average. Reflow replaces this pairing with the one the teacher defines: each noise sample is paired with the end point of its own ODE path. A new flow is then fitted to straight lines between these pairs. Its paths are nearly straight, so even one Euler step lands close to the end of the path.',
      'This step uses the 2D mixture of §2.2, where paths can cross. The teacher is exact, solved with RK4. The reflowed field is not available in closed form and is estimated by kernel regression on 4000 coupled pairs, so it carries a small bias. Switch between the teacher and the reflowed field and move K. The grey lines join each sample to the teacher end point of the same noise.',
    ],
    tex: `${v}(${x}, ${t}) = \\E\\big[\\x_1 - \\x_0 \\,\\big|\\, \\x_t = ${x}\\big], \\qquad (\\x_0, \\x_1) = \\big(\\x_0,\\ X_{0\\to 1}(\\x_0)\\big)`,
    after: ['The straightness of the teacher, the mean squared difference between the velocity on the path and the straight displacement, is 1.072; after one reflow it is 0.0012. With one Euler step the samples of the teacher field end on average 1.610 away from the end of their own path, and those of the reflowed field 0.087. With 2 and 4 steps the teacher needs 0.701 and 0.349, the reflowed field 0.067 and 0.056. The reflowed field does not reach zero because the kernel estimate smooths the field: following it with the ODE ends 0.047 away on average.'],
    check: {
      q: 'Why does one reflow already give straight, non-crossing paths in 1D?',
      opts: [
        ['The teacher coupling in 1D is monotone, so it is the optimal transport coupling of §2.6. The straight lines between its pairs never cross, and the reflowed flow follows them exactly.', true],
        ['In 1D the velocity field is constant in time.', false],
        ['In 1D the flow matching paths are already straight before any reflow.', false],
      ],
      fb: { true: 'Right. In 1D the increasing coupling is the optimal one for the squared cost, and the straight lines between increasing pairs do not cross. In 2D the teacher coupling is not optimal in general, which is why reflow can be repeated there.', false: 'Look at step 1: the 1D flow map keeps the order of points. What does that say about the straight lines between the pairs?' },
    },
  },
  {
    id: 'meanflow', kicker: 'Learning the jump', title: 'Average velocities', scene: 'meanflow',
    prose: [
      'Instead of composing small steps, a model can learn the jump itself, written as an average velocity: the displacement from s to t divided by the elapsed time. A single evaluation of this average velocity gives an exact jump of any length. As t approaches s it becomes the instantaneous velocity, so the model contains the teacher.',
      'MeanFlow trains such a model without a teacher solver. It uses an identity that links the average velocity to the instantaneous one and to the change of the average velocity along the path. Both sides can be evaluated with one forward pass and one Jacobian-vector product. Move s and t: on the left the chord of the path, whose slope is the average velocity, against the tangent at s; on the right the average velocity along six paths as a function of the end time.',
    ],
    tex: `${ub}(\\x_{${s}}, ${s}, ${t}) = ${u.replace('_t', '_s')}(\\x_{${s}}) + (${t} - ${s})\\,\\frac{\\dd}{\\dd ${s}}\\,${ub}(\\x_{${s}}, ${s}, ${t})`,
    deriv: {
      have: ['The definition \\((t - s)\\,\\ubar(\\x_s, s, t) = \\int_s^t \\vel_\\tau(\\x_\\tau)\\,\\dd\\tau\\), with \\(\\x_\\tau\\) on one ODE path.', 'The derivative d/ds is taken along the path, with t fixed.'],
      goal: '\\ubar = \\vel_s(\\x_s) + (t - s)\\,\\tfrac{\\dd}{\\dd s}\\ubar',
      plan: 'Differentiate both sides of the definition with respect to s.',
      title: 'The MeanFlow identity', lines: [
        { tex: `\\frac{\\dd}{\\dd s}\\Big[(t - s)\\,\\ubar(\\x_s, s, t)\\Big] = \\frac{\\dd}{\\dd s}\\int_s^t \\vel_\\tau(\\x_\\tau)\\,\\dd\\tau`, why: 'Both sides of the definition are functions of s along the path.', uses: [] },
        { tex: `${CHG('-\\ubar + (t - s)\\,\\frac{\\dd}{\\dd s}\\ubar')} = \\frac{\\dd}{\\dd s}\\int_s^t \\vel_\\tau(\\x_\\tau)\\,\\dd\\tau`, why: 'The product rule on the left.', uses: ['chain-rule'] },
        { tex: `-\\ubar + (t - s)\\,\\frac{\\dd}{\\dd s}\\ubar = ${CHG('-\\vel_s(\\x_s)')}`, why: 'The fundamental theorem of calculus: the derivative of an integral in its lower limit is minus the integrand there.', uses: ['ftc'] },
        { tex: `\\Rightarrow\\ \\ubar = ${CHG('\\vel_s(\\x_s) + (t - s)\\,\\frac{\\dd}{\\dd s}\\ubar')}`, why: 'Rearrange.', uses: [] },
      ],
    },
    after: ['At s = 0.2 and t = 0.7, on the marked path through 0.212, the tangent has slope −0.286 but the chord has slope 0.064: the path first moves down and then turns up towards the middle mode. The identity gives −0.286 + 0.5 × 0.701 = 0.064 on the right-hand side. Samples made with 1, 2 or 4 exact jumps all have W1 = 0.0009, the floor, because exact jumps compose without error. A learned average velocity has errors, and then more jumps can help, as with the two-step consistency samples.'],
    check: {
      q: 'Take s = 0 and x = 0. What value does \\(\\ubar(0, 0, t)\\) tend to as \\(t \\to 0\\)? Give three decimals.',
      answer: -0.155, tol: 0.005,
      fb: { true: 'Right. As the jump shrinks, the average velocity becomes the instantaneous velocity \\(\\vel_0(0) = -0.155\\). This is why an average-velocity model also contains the teacher field.', false: 'Divide the jump by its length and let the length go to zero. Step 1 told you what the derivative of the jump is at t = s. You can read \\(\\vel_0\\) from the readout of step 1 with s = 0, or set s = 0 here and move t down.' },
    },
  },
  {
    id: 'dmd', kicker: 'One step without pairs', title: 'Matching distributions', scene: 'dmd',
    prose: [
      'The students so far copy the teacher sample by sample. Distribution matching distillation (DMD) trains a one-step generator to match the distribution of the teacher instead. It minimizes the reverse KL from the generator to the data, averaged over noise levels. The gradient needs two scores at the noised generator samples: the score of the real data, which the teacher provides, and the score of the generator distribution, which a second model tracks during training. Their difference pushes each sample towards regions where the data are denser than the generator.',
      'Here both scores are exact. The generator is 400 free particles at fixed noise quantiles, started on the right mode, so its noised distribution is a Gaussian mixture with an exact score. The buttons choose which noise levels enter the loss and whether a regression loss on teacher pairs is added, as in the original DMD. Press Play and watch the share of the samples in each mode.',
    ],
    tex: `\\nabla_\\theta\\,\\E_t\\,\\KL\\big(p^{${gth}}_t\\,\\|\\,p_t\\big) = \\E_{t,\\,z,\\,\\varepsilon}\\Big[w_t\\,\\big(${sf}(\\x_t, t) - ${sr}(\\x_t, t)\\big)\\,\\frac{\\partial \\x_t}{\\partial\\theta}\\Big], \\quad \\x_t = t\\,${gth}(z) + (1 - t)\\,\\varepsilon`,
    deriv: {
      have: ['A generator \\(G_\\theta\\) whose samples \\(\\x_t = t\\,G_\\theta(z) + (1 - t)\\,\\varepsilon\\) have density \\(q_t\\).', 'The data at the same noise level have density \\(p_t\\).'],
      goal: '\\nabla_\\theta \\KL(q_t\\,\\|\\,p_t) = \\E\\big[(\\nabla\\log q_t - \\nabla\\log p_t)(\\x_t)\\,\\partial_\\theta \\x_t\\big]',
      plan: 'Write the KL as an expectation over the reparametrized sample, differentiate, and show that the derivative of q_t in θ at a fixed point contributes nothing.',
      title: 'The gradient of the reverse KL', lines: [
        { tex: `\\KL(q_t\\,\\|\\,p_t) = \\E_{z,\\varepsilon}\\big[\\log q_t(\\x_t) - \\log p_t(\\x_t)\\big]`, why: 'The definition of the KL, with \\(\\x_t\\) drawn by reparametrization.', uses: ['kl', 'expectation'] },
        { tex: `\\nabla_\\theta \\KL = \\E\\big[${CHG('(\\nabla\\log q_t - \\nabla\\log p_t)(\\x_t)\\,\\partial_\\theta\\x_t')}\\big] + ${CHG('\\E\\big[(\\partial_\\theta\\log q_t)(\\x_t)\\big]')}`, why: 'The chain rule through \\(\\x_t\\), plus the explicit dependence of \\(q_t\\) on θ at a fixed point.', uses: ['chain-rule', 'diff-under-int'] },
        { tex: `${CHG('\\E\\big[(\\partial_\\theta\\log q_t)(\\x_t)\\big] = \\int \\partial_\\theta q_t\\,\\dd\\x = \\partial_\\theta 1 = 0')}`, why: 'The expectation of a score in the parameters is zero, because \\(q_t\\) integrates to 1 for every θ.', uses: ['log-deriv', 'diff-under-int'] },
        { tex: `\\Rightarrow\\ \\nabla_\\theta \\KL = \\E\\big[(${CHG('s_{\\text{fake}} - s_{\\text{real}}')})(\\x_t)\\,\\partial_\\theta\\x_t\\big]`, why: 'Name the two scores. The weight \\(w_t\\) of the formula averages this over t; here \\(w_t = (1 - t)^2\\).', uses: [] },
      ],
    },
    after: ['With low noise only (t between 0.85 and 0.98) and no regression, the cloud spreads slowly from the right mode into the middle one, and after 400 iterations the shares are 0.06, 0.36 and 0.58 against 0.45, 0.30 and 0.25 in the data, with W1 = 1.336. At low noise the modes barely overlap, so the score difference is nearly zero far from the samples, and the left mode, which the generator does not cover, sends almost no signal. With the wide range of t (0.2 to 0.98) the high noise levels connect the modes and the shares reach 0.43, 0.32 and 0.24, with W1 = 0.079 after 200 and 0.015 after 400 iterations. The regression weight λ = 0.02 fixes the low-noise run as well: W1 = 0.029 after 200 iterations.'],
    check: {
      q: 'Why does the reverse KL, estimated at low noise only, leave a mode empty for so long, and what does the regression loss add?',
      opts: [
        ['The gradient is evaluated only at the generator samples. Where the generator has no mass, nothing pulls samples in, and at low noise the scores near the samples do not see the distant mode. The regression loss pairs each noise with a teacher end point, which assigns some samples to every mode.', true],
        ['The real score is zero in the empty mode, so the loss cannot be reduced there.', false],
        ['The regression loss makes the generator deterministic, which the reverse KL cannot do.', false],
      ],
      fb: { true: 'Right. The reverse KL is an expectation under the generator, so it only measures where the generator already is. That is the mode-seeking behaviour. A wider range of noise levels and the regression loss are the two remedies used in practice.', false: 'Look at which samples the expectation in the gradient runs over. Can a region with no generator samples contribute a gradient?' },
    },
    closing: `<h3>Where this leads</h3>
      <p>Every student on this page approximates the same object, the flow map of the teacher. The table collects W1 to the data for 1, 2 and 4 network evaluations. The progressive and consistency students are piecewise-linear maps with 16 knots per step or per time. The DMD generator is 400 free particles trained for 400 iterations on the wide range of t with λ = 0.02, so it is more flexible than the others and the numbers are not at equal capacity.</p>
      <table class="tbl"><thead><tr><th>Method</th><th>1 step</th><th>2 steps</th><th>4 steps</th></tr></thead><tbody>
        <tr><td>Euler on the exact field</td><td>1.310</td><td>0.628</td><td>0.320</td></tr>
        <tr><td>Progressive distillation from 64 Euler steps</td><td>0.026</td><td>0.027</td><td>0.027</td></tr>
        <tr><td>Consistency distillation, N = 64</td><td>0.026</td><td>0.035</td><td></td></tr>
        <tr><td>Consistency training, N = 64</td><td>0.067</td><td>0.050</td><td></td></tr>
        <tr><td>DMD with regression</td><td>0.004</td><td></td><td></td></tr>
        <tr><td>Exact flow map</td><td>0.0009</td><td>0.0009</td><td>0.0009</td></tr>
      </tbody></table>
      <p>The flow map and its velocity limit come from <a href="../flow-matching/">Flow matching</a> (§2.4), and the straight couplings of reflow from <a href="../couplings/">Couplings, transport and control</a> (§2.6). <a href="../inference-time/">Inference-time scaling</a> (§6) asks the opposite question: what extra network evaluations buy when they are spent at sampling time. Animations <a href="../../animations/flow-map-zoo/index.html">A9, the flow map zoo</a>, <a href="../../animations/reflow/index.html">A10, reflow</a>, and <a href="../../animations/dmd/index.html">A11, distribution matching</a>, show the same ideas in 2D.</p>`,
  },
];
