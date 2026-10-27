// Content of the §2.2 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const sx = `${S('stheta')}(${S('x')})`;
const pc = `${S('pcond')}`;

export const INTRO = `
  <div class="kicker">Survey §2.2 · Interactive tutorial</div>
  <h1>Learning the score: from Hyvärinen to Tweedie</h1>
  <p>The first tutorial assumed that the score was known. Annealed Langevin dynamics then turned noise into samples of \\(p\\), given the score of the smoothed density \\(p_\\sigma\\) at every level of a noise ladder. In practice \\(p\\) is unknown, and all we have is a dataset of samples from it. This tutorial shows how a model learns those scores from the dataset alone.</p>
  <p>A dataset allows one operation: averaging over it. A training objective must therefore be an expectation under \\(p\\), and the quantity inside the expectation must be computable from one sample and the model. Steps 1, 2 and 5 test three objectives against these two requirements. Explicit score matching fails the second. Implicit score matching passes, but at a cost that grows with the dimension. Denoising score matching passes cheaply and learns all noise levels with one network. Steps 3 and 4 build the pieces it needs, and steps 6 and 7 show what else the learned score gives.</p>
  <p>Each step states one idea, shows one formula and gives you a plot that the formula controls. Every symbol can be hovered or tapped, and the button "Symbols" in the top bar opens the full list.</p>
  <p>The data are Gaussian mixtures, so every score, posterior and denoiser on this page is exact. The interpolation is \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) with \\(\\alpha_t = t\\) and \\(\\sigma_t = 1 - t\\): \\(t = 0\\) is noise and \\(t = 1\\) is data.</p>
  <ol class="toc">
    <li><a href="#esm">Matching the score directly</a></li>
    <li><a href="#ism">Integration by parts removes the unknown score</a></li>
    <li><a href="#cond">The score of one noisy data point</a></li>
    <li><a href="#marginal">The marginal score is an average of conditional scores</a></li>
    <li><a href="#dsm">Regressing on a noisy target</a></li>
    <li><a href="#tweedie">Tweedie's formula: a denoiser is a score</a></li>
    <li><a href="#targets">Four interchangeable prediction targets</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'esm', kicker: 'Explicit score matching', title: 'Matching the score directly', scene: 'esm',
    prose: [
      'In §2.1 the score was given by a formula. Here we pretend it is not, and keep only the 200 samples drawn as dots under the axis. The two requirements of the introduction apply: the objective must be an average over these samples, and each term of the average must be computable.',
      'The most direct candidate compares the model score \\(\\s_\\theta\\) with the true score at every point. The squared distance is the natural way to compare two vector fields. Averaging it under \\(p\\) is what turns it into an expectation, and it also means that errors count only where the data lie. The result is the Fisher divergence, called explicit score matching. The derivation below shows why driving it to zero is enough: a model with the right score has the right density.',
      'On this page the model is the score of a single Gaussian \\(\\N(m, v)\\), a straight line \\(\\s_\\theta(x) = -(x-m)/v\\). Move \\(m\\) and \\(v\\). The lower panel shades the integrand, which is large where the line misses the dashed true score and \\(p\\) has mass. Its area is \\(J_{\\mathrm{ESM}}\\).',
    ],
    tex: `${S('J')}(${S('theta')}) = \\tfrac12\\,${S('Ep')}\\big\\|${sx} - ${S('glp')}(${S('x')})\\big\\|^2`,
    deriv: {
      title: 'Why matching the score is enough',
      have: ['A model density \\(p_\\theta\\) whose score is \\(\\s_\\theta = \\nabla\\log p_\\theta\\). As in §2.1, its normalizing constant does not enter the score.', 'A data density \\(p\\) that is positive everywhere, as the mixture here is.'],
      goal: `${S('J')}(${S('theta')}) = 0 \\;\\Longrightarrow\\; p_${S('theta')} = ${S('p')}`,
      plan: 'A zero average of a non-negative quantity forces the quantity to vanish. Equal scores mean equal log densities up to a constant, and normalization fixes the constant.',
      lines: [
      { tex: `${S('J')}(${S('theta')}) = 0 \\;\\Longrightarrow\\; ${sx} = ${S('glp')}(${S('x')}) \\text{ for all } ${S('x')}`, why: 'The integrand is a square times \\(p > 0\\). Its average is zero only if it vanishes everywhere, for continuous scores.', uses: ['expectation'] },
      { tex: `\\nabla\\big(${CHG(`\\log p_${S('theta')}(${S('x')}) - \\log ${S('p')}(${S('x')})`)}\\big) = 0`, why: 'Write \\(\\s_\\theta = \\nabla\\log p_\\theta\\) and move both terms under one gradient.', uses: ['grad-linear'] },
      { tex: `\\log p_${S('theta')}(${S('x')}) = \\log ${S('p')}(${S('x')}) + ${CHG('c')}`, why: 'A function with zero gradient everywhere is constant.' },
      { tex: `p_${S('theta')} = ${CHG('e^{c}')}\\,${S('p')}, \\qquad 1 = \\int p_${S('theta')}\\,\\dd${S('x')} = e^{c}\\!\\int ${S('p')}\\,\\dd${S('x')} = e^{c} \\;\\Longrightarrow\\; ${CHG('c = 0')}`, why: 'Exponentiate, then integrate both sides. Both densities integrate to one, so \\(e^c = 1\\) and the densities agree.' },
    ] },
    after: ['So \\(J_{\\mathrm{ESM}}\\) is the right target, and it passes the first requirement: it is an average under \\(p\\). It fails the second, because each term contains \\(\\nabla\\log p\\), the very function we want to learn, and the samples do not tell us its value. (A straight line cannot follow a three-mode score, so \\(J_{\\mathrm{ESM}}\\) never reaches zero on this page. The best line is still well defined.)'],
    check: {
      q: 'Where does a missed score cost the most in \\(J_{\\mathrm{ESM}}\\)?',
      opts: [['Where the gap between the two curves is largest', false], ['Where the gap is large and p has mass', true], ['In the tails, where p is small', false]],
      fb: { true: 'Right. The squared gap is multiplied by p. In the tails the gap can be huge and still cost almost nothing, which is why score models are poor far from the data.', false: 'Look at the lower panel. The squared gap is multiplied by p(x), so a large miss in the tails, where p is nearly zero, adds almost nothing to the area.' },
    },
  },
  {
    id: 'ism', kicker: 'Implicit score matching', title: 'Integration by parts removes the unknown score', scene: 'ism',
    prose: [
      'Hyvärinen (2005) showed that the unknown score can be removed, which fixes the second requirement. Expanding the square gives a cross term \\(\\E_p[\\s_\\theta\\cdot\\nabla\\log p]\\), and integration by parts turns it into \\(-\\E_p[\\diver\\s_\\theta]\\), which needs only the model. Step through the derivation below.',
      'The result differs from \\(J_{\\mathrm{ESM}}\\) by a constant \\(C\\) that does not depend on \\(\\theta\\). Move \\(m\\) and \\(v\\) and watch both readouts: they change together, and their difference stays fixed. Then press "Fit from samples". It runs gradient descent on an estimate of the new objective from the 200 samples on the axis, and never evaluates \\(p\\) or its score.',
    ],
    tex: `${S('J')}(${S('theta')}) = ${S('Ep')}\\Big[\\tfrac12\\|${sx}\\|^2 + ${S('div')}${sx}\\Big] + ${S('C')}`,
    deriv: {
      have: ['The explicit loss \\(J(\\theta) = \\tfrac12\\E_p\\|\\s_\\theta(\\x) - \\nabla\\log p(\\x)\\|^2\\) of step 1. It contains the unknown score.', 'Samples from p, but not p itself.'],
      goal: 'step',
      plan: 'Expand the square. Only the cross term contains both the model and the unknown score, and integration by parts moves the derivative from p onto the model.',
      lines: [
      { tex: `${S('J')} = \\tfrac12${S('Ep')}\\|${S('stheta')}\\|^2 - ${S('Ep')}\\big[${S('stheta')}\\cdot${S('glp')}\\big] + \\tfrac12${S('Ep')}\\|${S('glp')}\\|^2`, why: 'Expand the square \\(\\|a-b\\|^2 = \\|a\\|^2 - 2a\\cdot b + \\|b\\|^2\\) and take the expectation term by term.', uses: ['expand-square', 'expectation'] },
      { tex: `${S('Ep')}\\big[${S('stheta')}\\cdot${S('glp')}\\big] = \\int ${S('p')}\\,${S('stheta')}\\cdot${CHG(`\\frac{\\nabla ${S('p')}}{${S('p')}}`)}\\,\\dd${S('x')} = \\int ${S('stheta')}\\cdot\\nabla ${S('p')}\\,\\dd${S('x')}`, why: 'Write the expectation as an integral and use \\(\\nabla\\log p = \\nabla p/p\\). The density cancels.', uses: ['expectation', 'log-deriv'] },
      { tex: `= ${CHG(`-\\int ${S('p')}\\,${S('div')}${S('stheta')}\\,\\dd${S('x')}`)} = -${S('Ep')}\\big[${S('div')}${S('stheta')}\\big]`, why: 'Integrate by parts. The boundary term vanishes when \\(p\\,\\s_\\theta\\) goes to zero at infinity.', uses: ['ibp'] },
      { tex: `${S('J')} = ${S('Ep')}\\Big[\\tfrac12\\|${S('stheta')}\\|^2 ${CHG(`+ ${S('div')}${S('stheta')}`)}\\Big] + ${CHG(`${S('C')}`)},\\quad ${S('C')} = \\tfrac12${S('Ep')}\\|${S('glp')}\\|^2`, why: 'Insert the cross term into the first line. The last term of the first line does not contain θ, so it becomes the constant C.' },
    ] },
    after: ['For the Gaussian model, \\(\\diver\\s_\\theta = -1/v\\) everywhere, and the minimizer is the sample mean and variance. The fit lands there. Implicit score matching therefore passes both requirements. Its cost is the divergence: for a neural network in \\(d\\) dimensions, it is the trace of a \\(d\\times d\\) Jacobian, needed again for every sample.'],
    check: {
      q: 'Computing \\(\\diver\\s_\\theta\\) exactly for one image with \\(d = 10^5\\) pixels takes how many backward passes through the network?',
      opts: [['One', false], ['About d', true], ['About d²', false]],
      fb: { true: 'Right. Each backward pass gives one row of the Jacobian, and the trace needs all d diagonal entries. This cost is what denoising score matching avoids.', false: 'One backward pass gives the gradient of one output coordinate, which is one row of the Jacobian. The trace needs d diagonal entries from d different rows, so it costs about d passes.' },
    },
  },
  {
    id: 'cond', kicker: 'Adding noise', title: 'The score of one noisy data point', scene: 'cond',
    prose: [
      'Vincent (2011) avoided the Jacobian by adding noise to the data. There are two reasons to do so. This step shows the first: with noise, the target becomes computable. The second comes from §2.1. Annealed Langevin dynamics never used the score of \\(p\\) itself. It used the scores of the smoothed densities \\(p_\\sigma\\) at a whole ladder of noise levels, so those are the scores we need to learn anyway.',
      'From here on the noisy point is written \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), with a data point \\(\\x_1\\) and noise \\(\\x_0 \\sim \\N(0, I)\\). Dividing by \\(\\alpha_t\\) gives \\(\\x_t/\\alpha_t = \\x_1 + (\\sigma_t/\\alpha_t)\\,\\x_0\\), which is the noisy point of §2.1 with \\(\\sigma = \\sigma_t/\\alpha_t\\). Each time \\(t\\) is therefore one rung of the noise ladder, and \\(p_t\\) is \\(p_\\sigma\\) shrunk by the factor \\(\\alpha_t\\). Here \\(t = 1\\) is the data and \\(t \\to 0\\) is pure noise.',
      'Now fix one data point \\(\\x_1\\). The noisy point \\(\\x_t\\) is then Gaussian around \\(\\alpha_t\\x_1\\) with standard deviation \\(\\sigma_t\\), and the score of a Gaussian is known in closed form.',
      'Click in the plot to choose \\(\\x_1\\), and move \\(t\\). Every arrow is the conditional score at one noisy point. It points back to \\(\\alpha_t\\x_1\\), and its length is \\(\\|\\x_0\\|/\\sigma_t\\): it depends only on the noise draw. As \\(t \\to 1\\) the circles shrink and the arrows grow without bound.',
    ],
    tex: `\\begin{aligned} ${pc} &= \\N\\big(${S('x')};\\,${S('alpha')}${S('x1')},\\,${S('sig')}^2 I\\big) \\\\ ${S('cs')} &= -\\frac{${S('xt')} - ${S('alpha')}${S('x1')}}{${S('sig')}^2} = -\\frac{${S('x0')}}{${S('sig')}} \\end{aligned}`,
    deriv: {
      have: ['The interpolation \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) with \\(\\x_0\\sim\\N(0, I)\\). Given \\(\\x_1\\), it is Gaussian with mean \\(\\alpha_t\\x_1\\) and covariance \\(\\sigma_t^2 I\\).'],
      goal: `${S('cs')} = -\\frac{${S('x0')}}{${S('sig')}}`,
      plan: 'Write the log density of this Gaussian, take its gradient in \\(\\x\\), and evaluate it at the noisy sample \\(\\x_t\\).',
      lines: [
      { tex: `\\log ${pc} = -\\frac{\\|${S('x')} - ${S('alpha')}${S('x1')}\\|^2}{2${S('sig')}^2} + \\text{const}`, why: 'Log of an isotropic Gaussian density. The constant collects the normalization, which does not depend on x.', uses: ['gauss', 'log-rules'] },
      { tex: `\\nabla_{${S('x')}}\\log ${pc} = ${CHG(`-\\frac{${S('x')} - ${S('alpha')}${S('x1')}}{${S('sig')}^2}`)}`, why: 'The gradient of \\(\\|\\x-\\bm c\\|^2/2\\) is \\(\\x - \\bm c\\).', uses: ['gauss'] },
      { tex: `${S('cs')} = -\\frac{${CHG(`${S('sig')}${S('x0')}`)}}{${S('sig')}^2} = -\\frac{${S('x0')}}{${S('sig')}}`, why: 'Evaluate at \\(\\x = \\x_t\\) and use \\(\\x_t - \\alpha_t\\x_1 = \\sigma_t\\x_0\\).' },
    ] },
    check: {
      q: 'At \\(t = 0.75\\), so \\(\\sigma_t = 0.25\\), a noise draw has \\(\\|\\x_0\\| = 1\\). How long is the conditional score at \\(\\x_t\\)?',
      answer: 4, tol: 0.01,
      fb: { true: 'Right. ‖x_0‖/σ_t = 1/0.25 = 4. Set t = 0.75: the readout gives the same value for the highlighted draw, which has ‖x_0‖ = 1.', false: 'Use the last form of the formula: the length is ‖x_0‖ divided by σ_t. With ‖x_0‖ = 1 and σ_t = 0.25 this is 1/0.25.' },
    },
  },
  {
    id: 'marginal', kicker: 'Marginal score', title: 'The marginal score is an average of conditional scores', scene: 'marginal',
    prose: [
      'Fix a noisy point \\(\\x\\). Many data points \\(\\x_1\\) could have produced it, each with a posterior probability given \\(\\x_t = \\x\\). Draw 60 such data points from that posterior. Each has its own conditional score, drawn as a thin arrow from \\(\\x\\) toward \\(\\alpha_t\\x_1\\), and a dot at \\(\\alpha_t\\x_1\\) coloured by its mixture component.',
      'The bold arrow is the average of the thin arrows. The dashed arrow is the exact marginal score \\(\\nabla\\log p_t(\\x)\\). Move the probe and \\(t\\): the two agree up to the error of averaging 60 draws. Near a mode at large \\(t\\), the posterior is narrow and the arrows agree. Between modes, they point in different directions and partly cancel.',
    ],
    tex: `\\begin{aligned} ${S('ms')} &= \\E\\big[${S('cs')}\\,\\big|\\,${S('xt')} = ${S('x')}\\big] \\\\ &= -\\frac{1}{${S('sig')}}\\,${S('Ex0')} \\end{aligned}`,
    deriv: {
      have: ['The marginal density as a mixture of conditionals, \\(p_t(\\x) = \\int p(\\x\\mid\\x_1)\\,\\pdata(\\x_1)\\,\\dd\\x_1\\).', 'The conditional score \\(-\\x_0/\\sigma_t\\) of step 3.'],
      goal: 'step',
      plan: 'Differentiate the mixture under the integral and divide by \\(p_t\\). The weights that appear are the posterior of \\(\\x_1\\) given \\(\\x_t\\), by Bayes’ rule.',
      lines: [
      { tex: `${S('pt')}(${S('x')}) = \\int ${pc}\\,p_{\\mathrm{data}}(${S('x1')})\\,\\dd${S('x1')}`, why: 'The marginal density averages the conditional densities over the data.', uses: ['bayes'] },
      { tex: `\\nabla ${S('pt')}(${S('x')}) = \\int ${CHG(`\\nabla\\log ${pc}\\;${pc}`)}\\,p_{\\mathrm{data}}(${S('x1')})\\,\\dd${S('x1')}`, why: 'Differentiate under the integral and write \\(\\nabla q = q\\nabla\\log q\\).', uses: ['diff-under-int', 'log-deriv'] },
      { tex: `${S('ms')} = \\int \\nabla\\log ${pc}\\;${CHG(`\\frac{${pc}\\,p_{\\mathrm{data}}(${S('x1')})}{${S('pt')}(${S('x')})}`)}\\,\\dd${S('x1')}`, why: 'Divide by \\(p_t\\)(x). The fraction is the posterior density of \\(\\x_1\\) given \\(\\x_t = \\x\\), by Bayes’ rule.', uses: ['bayes'] },
      { tex: `= ${CHG(`\\E\\big[${S('cs')}\\,\\big|\\,${S('xt')} = ${S('x')}\\big]`)} = -\\frac{1}{${S('sig')}}\\,${S('Ex0')}`, why: 'An integral against the posterior is a conditional expectation. Insert the conditional score \\(-\\x_0/\\sigma_t\\) from step 3.', uses: ['cond-exp', 'cond-score'] },
    ] },
    after: ['The marginal score at \\(\\x\\) is the average of the conditional scores of all data points that could have produced \\(\\x\\). The conditional scores are known, but the average needs the posterior, which needs \\(p_t\\). Step 5 shows that a regression computes this average for us, from samples alone.'],
  },
  {
    id: 'dsm', kicker: 'Denoising score matching', title: 'Regressing on a noisy target', scene: 'dsm',
    prose: [
      'Step 4 wrote the marginal score as an average, \\(\\nabla\\log p_t(\\x) = \\E[-\\x_0/\\sigma_t\\mid\\x_t = \\x]\\). We cannot compute this average directly, because it needs the posterior. A least-squares regression computes such conditional averages for us. Draw a data point \\(\\x_1\\) and a noise draw \\(\\x_0\\), form \\(\\x_t\\), and train the model to predict the label \\(-\\x_0/\\sigma_t\\) from the input \\(\\x_t\\). The label is known, because we drew the noise ourselves. This regression is denoising score matching.',
      'The plot shows 2500 such training pairs for the one-dimensional data of steps 1 and 2. Each dot is one pair, with its input \\(\\x_t\\) on the horizontal axis and its label on the vertical axis. Dots above the same input have very different labels, because the same \\(\\x_t\\) arises from many combinations of data point and noise. No model can predict the label of a single dot. A squared-error fit learns the mean label at each input instead, shown by the open circles, and the circles lie on the solid curve, the exact marginal score. Move the probe to see the mean and the spread of the labels near one input, and move \\(t\\) to see that this holds at every noise level.',
    ],
    tex: `\\begin{aligned} ${S('Jd')}(${S('theta')}) &= \\tfrac12\\,\\E_{${S('x1')},${S('x0')}}\\Big\\|${S('stheta')}(${S('xt')},${S('t')}) - \\Big(-\\frac{${S('x0')}}{${S('sig')}}\\Big)\\Big\\|^2 \\\\ {${S('stheta')}}^{\\star}(${S('x')},${S('t')}) &= \\E\\Big[-\\frac{${S('x0')}}{${S('sig')}}\\,\\Big|\\,${S('xt')} = ${S('x')}\\Big] = ${S('ms')} \\end{aligned}`,
    deriv: {
      have: ['Two random variables, an input X and a label Y, and any function f of X.', 'The conditional mean \\(\\E[Y\\mid X]\\): for each value of X, the average of the labels that occur with it. It is itself a function of X.', 'For the application: \\(X = \\x_t\\), \\(Y = -\\x_0/\\sigma_t\\) and, by step 4, \\(\\E[Y\\mid X] = \\nabla\\log p_t(\\x_t)\\).'],
      goal: `\\E\\|${S('f')}(${S('X')}) - ${S('Y')}\\|^2 = \\E\\|${S('f')}(${S('X')}) - ${S('EY')}\\|^2 + \\E\\|${S('Y')} - ${S('EY')}\\|^2`,
      plan: 'Split the error of f into its distance from the conditional mean and the distance of the label from the conditional mean. Expanding the square gives the two terms of the goal and a cross term. The cross term is zero, because once X is fixed the first factor is a constant and the second averages to zero. Finally put in the score-matching variables.',
      title: 'Proof of the regression lemma', lines: [
      { tex: `${S('f')}(${S('X')}) - ${S('Y')} = \\underbrace{\\big(${S('f')}(${S('X')}) - ${S('EY')}\\big)}_{${CHG('\\bm a')}} + \\underbrace{\\big(${S('EY')} - ${S('Y')}\\big)}_{${CHG('\\bm b')}}`, why: 'Add and subtract the conditional mean. Nothing changes yet, but the error now has two parts: \\(\\bm a\\), how far f is from the best prediction, and \\(\\bm b\\), how far the best prediction is from the label. Only \\(\\bm a\\) involves f, and \\(\\bm a\\) is a function of X alone.', uses: ['expand-square'] },
      { tex: `\\E\\|${S('f')}(${S('X')}) - ${S('Y')}\\|^2 = \\E\\|\\bm a\\|^2 + \\E\\|\\bm b\\|^2 + ${CHG('2\\,\\E[\\bm a\\cdot\\bm b]')}`, why: 'Square with \\(\\|\\bm a + \\bm b\\|^2 = \\|\\bm a\\|^2 + \\|\\bm b\\|^2 + 2\\,\\bm a\\cdot\\bm b\\) and average each term. The first two terms are those of the goal, so it remains to show that the cross term is zero.', uses: ['expand-square', 'expectation'] },
      { tex: `\\E[\\bm a\\cdot\\bm b] = \\E\\big[\\,${CHG(`\\E[\\bm a\\cdot\\bm b\\mid ${S('X')}]`)}\\,\\big]`, why: 'Average in two stages: first over the labels that occur with one value of X, then over X. This is the tower property and gives the same number.', uses: ['cond-exp'] },
      { tex: `= \\E\\big[\\bm a\\cdot ${CHG(`\\E[\\bm b\\mid ${S('X')}]`)}\\big]`, why: 'Once X is fixed, \\(\\bm a = f(X) - \\E[Y\\mid X]\\) is a fixed vector. It is a constant inside the inner average and can be taken out of it.', uses: ['cond-exp'] },
      { tex: `\\E[\\bm b\\mid ${S('X')}] = ${S('EY')} - \\E[${S('Y')}\\mid ${S('X')}] = ${CHG('0')}`, why: 'Inside the inner average, \\(\\E[Y\\mid X]\\) is a constant and stays as it is, while Y averages to \\(\\E[Y\\mid X]\\) by definition. The residual of the label has mean zero for every X, so the cross term vanishes, whatever f is.', uses: ['cond-exp'] },
      { tex: `\\E\\|${S('f')}(${S('X')}) - ${S('Y')}\\|^2 = ${CHG(`\\E\\|${S('f')}(${S('X')}) - ${S('EY')}\\|^2`)} + \\E\\|${S('Y')} - ${S('EY')}\\|^2`, why: 'Insert the zero into the second line. This is the lemma. Only the first term depends on f. It is never negative and is zero for \\(f = \\E[Y\\mid X]\\), so the conditional mean is the minimizer.' },
      { tex: `${S('Jd')}(${S('theta')}) = ${CHG(`\\tfrac12\\,\\E\\big\\|${S('stheta')}(${S('xt')},${S('t')}) - \\nabla\\log ${S('pt')}(${S('xt')})\\big\\|^2`)} + C_t`, why: 'Apply the lemma with \\(X = \\x_t\\), \\(f = \\s_\\theta(\\cdot, t)\\) and \\(Y = -\\x_0/\\sigma_t\\), and multiply by one half. By step 4 the conditional mean is the marginal score. The first term is the explicit objective of step 1, written for \\(p_t\\). The second term, \\(C_t\\), is half the spread of the labels around the marginal score. It does not contain \\(\\theta\\), so both objectives have the same gradient and the same minimizer.', uses: ['marginal-score'] },
    ] },
    after: ['Why does this regression learn the score? Every label is a noise draw that no model can predict, so the loss never reaches zero, and it is not obvious what a model that minimizes it has learned. Step 1 asked for something else, the explicit objective \\(J_{\\mathrm{ESM}}\\), which measures the distance to the true score and cannot be computed. We need to know that minimizing \\(J_{\\mathrm{DSM}}\\) gives the same model as minimizing \\(J_{\\mathrm{ESM}}\\). Both claims follow from one fact about least squares, the regression lemma proved below.', 'For any input \\(X\\) and label \\(Y\\), the squared error of a predictor \\(f\\) splits into two parts. The first is the distance of \\(f\\) from the conditional mean \\(\\E[Y\\mid X]\\), and the second is the spread of \\(Y\\) around that mean. Only the first part depends on \\(f\\), so the conditional mean is the best predictor, which is the second line of the formula. With \\(X = \\x_t\\) and \\(Y = -\\x_0/\\sigma_t\\), step 4 says that the conditional mean is the marginal score. The first part is then the explicit objective for the noised density \\(p_t\\), and the second part is a constant. As with the implicit objective of step 2, \\(J_{\\mathrm{DSM}}\\) equals \\(J_{\\mathrm{ESM}}\\) plus a constant, and the last line of the proof writes this out.', 'The readout checks this numerically. The loss of the exact score is not zero: it is the spread of the labels, the constant part. The loss of the predictor \\(f = 0\\) is larger by \\(\\tfrac12\\E\\|\\nabla\\log p_t\\|^2\\), which is the explicit objective of \\(f = 0\\), as the lemma says.', 'Denoising score matching passes both requirements of step 1. It is an average over data points and noise draws, and each term needs only the model and the noise that was added, with no divergence. Sampling \\(t\\) at random as well trains one network \\(\\s_\\theta(\\x, t)\\) for every noise level at once. The same pattern of a tractable conditional target with an intractable marginal minimizer returns in flow matching (§2.4), in distillation and in reward fine-tuning.'],
    check: {
      q: 'A perfect score model is trained with \\(J_{\\mathrm{DSM}}\\). What is its loss?',
      opts: [['Zero', false], ['The second term of the lemma, the spread of the labels around their mean', true], ['It cannot be computed', false]],
      fb: { true: 'Right. The readout shows it: the exact score still has a large loss, equal to the variance of the dots around the curve. This is why DSM loss values say little about model quality on their own.', false: 'Look at the readout "J_DSM of the exact score". Even the exact marginal score cannot predict which noise draw produced a given x_t, so the spread of the labels, the second term of the lemma, remains.' },
    },
  },
  {
    id: 'tweedie', kicker: 'Tweedie’s formula', title: 'Tweedie’s formula: a denoiser is a score', scene: 'tweedie',
    prose: [
      'The denoiser \\(\\E[\\x_1\\mid\\x_t = \\x]\\) is the best guess, in squared error, of the clean data point behind a noisy point \\(\\x\\). Tweedie’s formula computes it from the score alone.',
      'The idea is already in the plot of step 4. There, each conditional score, multiplied by \\(\\sigma_t^2\\), was an arrow from \\(\\x\\) to the point \\(\\alpha_t\\x_1\\) of one posterior draw, and the marginal score was the average of these arrows. The average arrow therefore ends at the average end point: \\(\\x + \\sigma_t^2\\nabla\\log p_t(\\x) = \\alpha_t\\,\\E[\\x_1\\mid\\x_t = \\x]\\). The formula makes two moves. It steps from \\(\\x\\) by \\(\\sigma_t^2\\) times the score, which lands at \\(\\alpha_t\\) times the denoised point, and it then divides by \\(\\alpha_t\\), which undoes the shrinkage toward the origin.',
      'The plot shows the two moves at the probe, and 300 noisy points, each joined to its denoised position. Move \\(t\\). Close to \\(t = 1\\) little noise has been added and each point barely moves. At intermediate \\(t\\) the denoised points gather at the centres of the modes. Close to \\(t = 0\\) they collapse onto the data mean, because a noisy point then says almost nothing about the data point that produced it.',
    ],
    tex: `${S('Ex1')} = \\frac{${S('x')} + ${S('sig')}^2\\,${S('ms')}}{${S('alpha')}}`,
    deriv: {
      have: ['The interpolation \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\).', 'The noise prediction of step 4, \\(\\E[\\x_0\\mid\\x_t = \\x] = -\\sigma_t\\nabla\\log p_t(\\x)\\).'],
      goal: 'step',
      plan: 'Take the conditional expectation of the interpolation given \\(\\x_t\\), insert the noise prediction, and solve for the data prediction.',
      lines: [
      { tex: `${S('x')} = ${S('alpha')}\\,${S('Ex1')} + ${S('sig')}\\,${S('Ex0')}`, why: 'Take the conditional expectation of \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) given \\(\\x_t = \\x\\). The left side is known given itself.', uses: ['cond-exp', 'expectation'] },
      { tex: `${S('x')} = ${S('alpha')}\\,${S('Ex1')} ${CHG(`- ${S('sig')}^2\\,${S('ms')}`)}`, why: 'Insert \\(\\E[\\x_0\\mid\\x_t=\\x] = -\\sigma_t\\nabla\\log p_t(\\x)\\) from step 4.', uses: ['marginal-score'] },
      { tex: `${S('Ex1')} = ${CHG(`\\frac{${S('x')} + ${S('sig')}^2\\,${S('ms')}}{${S('alpha')}}`)}`, why: 'Solve for the posterior mean.' },
    ] },
    after: ['The denoised point is an average, not a sample. When a noisy point could have come from two modes, its denoised point lies between them, where the data have little mass. This is why a single denoising step from large noise gives a blurry result, and why a sampler cannot jump from pure noise to the denoiser in one step.', 'The denoiser and the score carry the same information. A network trained to denoise is a score model, and a score model is a denoiser. The survey uses this identity for posterior sampling in inverse problems (§6), for distribution-matching distillation (§5) and for representation alignment (§2.8).'],
    check: {
      q: 'As \\(t\\) approaches 0, where does \\(\\E[\\x_1\\mid\\x_t]\\) go for every noisy point?',
      opts: [['It stays at the noisy point', false], ['To the nearest mode', false], ['To the mean of the data', true]],
      fb: { true: 'Right. At t = 0 the noisy point is pure noise and carries no information about x_1, so the best guess is the unconditional mean. Move t toward 0 and the filled dots meet at the cross.', false: 'At t = 0 the noisy point is pure noise, independent of x_1. The posterior is the prior, so the best guess in squared error is the data mean. Move t toward 0 and watch the filled dots.' },
    },
  },
  {
    id: 'targets', kicker: 'Parametrizations', title: 'Four interchangeable prediction targets', scene: 'targets',
    prose: [
      'Steps 4 to 6 met four quantities that a network could output at a noisy point \\(\\x\\): the denoiser \\(\\hx_1 = \\E[\\x_1\\mid\\x_t = \\x]\\), the noise prediction \\(\\hx_0 = \\E[\\x_0\\mid\\x_t = \\x]\\), the score \\(\\s_t\\), and the velocity \\(\\hat{\\vel}_t\\) that flow matching (§2.4) learns. Taking the conditional expectation of \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) given \\(\\x_t = \\x\\) gives \\(\\x = \\alpha_t\\hx_1 + \\sigma_t\\hx_0\\). Since \\(\\x\\) is known, any one of the four determines the other three. The table writes each of them in terms of \\(\\hx_1\\).',
      `<div class="cmp-wrap"><table class="cmp">
        <thead><tr><th>Output</th><th>Common name</th><th>In terms of \\(\\hx_1\\)</th><th>Weight on \\(\\|\\hx_1 - \\x_1\\|^2\\)</th><th>Weight, linear path</th><th>Used by</th></tr></thead>
        <tbody>
          <tr><td>\\(\\hx_1\\)</td><td>data prediction, denoiser</td><td>\\(\\hx_1\\)</td><td>\\(1\\)</td><td>\\(1\\)</td><td><a href="https://arxiv.org/abs/2511.13720">JiT</a> (Li and He, 2025); the denoiser \\(D_\\theta\\) of <a href="https://arxiv.org/abs/2206.00364">EDM</a> (Karras et al., 2022)</td></tr>
          <tr><td>\\(\\hx_0\\)</td><td>noise or ε-prediction</td><td>\\((\\x - \\alpha_t\\hx_1)/\\sigma_t\\)</td><td>\\(\\alpha_t^2/\\sigma_t^2\\)</td><td>\\(t^2/(1-t)^2\\)</td><td><a href="https://arxiv.org/abs/2006.11239">DDPM</a> (Ho et al., 2020), see the <a href="../ddpm/">DDPM tutorial</a></td></tr>
          <tr><td>\\(\\s_t\\)</td><td>score</td><td>\\(-(\\x - \\alpha_t\\hx_1)/\\sigma_t^2\\)</td><td>\\(\\alpha_t^2/\\sigma_t^4\\)</td><td>\\(t^2/(1-t)^4\\)</td><td><a href="https://arxiv.org/abs/1907.05600">NCSN</a> (Song and Ermon, 2019), steps 4 to 6 of this page</td></tr>
          <tr><td>\\(\\hat{\\vel}_t\\)</td><td>velocity</td><td>\\((\\dot\\alpha_t - \\dot\\sigma_t\\alpha_t/\\sigma_t)\\,\\hx_1 + (\\dot\\sigma_t/\\sigma_t)\\,\\x\\)</td><td>\\((\\dot\\alpha_t - \\dot\\sigma_t\\alpha_t/\\sigma_t)^2\\)</td><td>\\(1/(1-t)^2\\)</td><td><a href="https://arxiv.org/abs/2210.02747">flow matching</a> (Lipman et al., 2023) and <a href="https://arxiv.org/abs/2209.03003">rectified flow</a> (Liu et al., 2023), see the <a href="../flow-matching/">flow-matching tutorial</a></td></tr>
        </tbody></table></div>`,
      'The last column names a model family that trains each output. The score was the original target of score matching. DDPM predicts the noise, and its loss is the noise loss of this table with a constant weight. Flow matching and rectified flow regress the velocity. Data prediction is not new either: Ho et al. (2020) tried it and found that noise prediction gave better samples. EDM trains a denoiser \\(D_\\theta\\) with a data loss, although its raw network output is a rescaled mixture of data and noise, and JiT returns to a network that outputs \\(\\hx_1\\) directly. The models differ in their output and their weighting, but by this step they learn the same posterior means.',
      'On the linear path the relation reads \\(\\x = t\\,\\hx_1 + (1-t)\\,\\hx_0\\), so the probe \\(\\x\\) lies on the segment from \\(\\hx_0\\) to \\(\\hx_1\\), a fraction \\(t\\) of the way along. The velocity \\(\\hat{\\vel}_t = \\hx_1 - \\hx_0\\) is that segment, drawn as an arrow. The score, multiplied by \\(\\sigma_t^2\\), is the arrow from \\(\\x\\) to \\(\\alpha_t\\hx_1\\), as in steps 4 and 6. Move the probe and \\(t\\).',
    ],
    tex: `\\begin{aligned} ${S('x')} &= ${S('alpha')}\\,${S('hx1')} + ${S('sig')}\\,${S('hx0')}, \\qquad ${S('st')} = -\\frac{${S('hx0')}}{${S('sig')}} \\\\ ${S('hu')} &= ${S('adot')}\\,${S('hx1')} + ${S('sdot')}\\,${S('hx0')} \\\\ \\|${S('hu')} - \\vel_t\\|^2 &= \\Big(${S('adot')} - ${S('sdot')}\\frac{${S('alpha')}}{${S('sig')}}\\Big)^2\\,\\|${S('hx1')} - ${S('x1')}\\|^2 \\end{aligned}`,
    deriv: {
      have: ['For the outputs: \\(\\x = \\alpha_t\\hx_1 + \\sigma_t\\hx_0\\), \\(\\s_t = -\\hx_0/\\sigma_t\\) and \\(\\hat{\\vel}_t = \\dot\\alpha_t\\hx_1 + \\dot\\sigma_t\\hx_0\\). Here \\(\\hx_1\\) is any output of the network, not only the optimum.', 'For a training pair with \\(\\x_t = \\x\\): \\(\\x = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), with labels \\(\\x_1\\), \\(\\x_0\\), \\(-\\x_0/\\sigma_t\\) and \\(\\vel_t = \\dot\\alpha_t\\x_1 + \\dot\\sigma_t\\x_0\\).'],
      goal: 'step',
      plan: 'Subtract the two relations for \\(\\x\\). This ties the noise error to the data error. The score and velocity errors are linear in these two, so they follow.',
      title: 'Derivation of the weights', lines: [
      { tex: `${S('alpha')}\\,(${S('hx1')} - ${S('x1')}) + ${S('sig')}\\,(${S('hx0')} - ${S('x0')}) = 0`, why: 'Subtract \\(\\x = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) from \\(\\x = \\alpha_t\\hx_1 + \\sigma_t\\hx_0\\).' },
      { tex: `${S('hx0')} - ${S('x0')} = ${CHG(`-\\frac{${S('alpha')}}{${S('sig')}}\\,(${S('hx1')} - ${S('x1')})`)}`, why: 'Solve for the noise error. Its square carries the weight \\(\\alpha_t^2/\\sigma_t^2\\).' },
      { tex: `${S('st')} - \\Big(-\\frac{${S('x0')}}{${S('sig')}}\\Big) = -\\frac{${S('hx0')} - ${S('x0')}}{${S('sig')}} = ${CHG(`\\frac{${S('alpha')}}{${S('sig')}^2}\\,(${S('hx1')} - ${S('x1')})`)}`, why: 'The score is the noise prediction divided by \\(-\\sigma_t\\), and its label is the conditional score of step 3. Its square carries the weight \\(\\alpha_t^2/\\sigma_t^4\\).', uses: ['cond-score'] },
      { tex: `${S('hu')} - \\vel_t = ${S('adot')}\\,(${S('hx1')} - ${S('x1')}) + ${S('sdot')}\\,(${S('hx0')} - ${S('x0')}) = ${CHG(`\\Big(${S('adot')} - ${S('sdot')}\\frac{${S('alpha')}}{${S('sig')}}\\Big)(${S('hx1')} - ${S('x1')})`)}`, why: 'Subtract the label from the output and insert the second line. Squaring gives the last line of the formula.' },
      { tex: `= ${CHG(`\\frac{1}{1-${S('t')}}`)}\\,(${S('hx1')} - ${S('x1')})`, why: 'For the linear path, \\(\\dot\\alpha_t = 1\\) and \\(\\dot\\sigma_t = -1\\), so the bracket is \\(1 + t/(1-t) = 1/(1-t)\\).' },
    ] },
    after: ['This is why the choice of output matters even though all four are equivalent. Each output is trained by regressing on its own label. For a fixed input \\(\\x\\), every training pair satisfies \\(\\alpha_t\\x_1 + \\sigma_t\\x_0 = \\x\\), and every set of outputs satisfies the same relation. An error in the data prediction therefore comes with a fixed multiple of itself in every other output, and the four squared losses differ only by the factors in the fourth column of the table.', 'At the optimum the factors do not matter, because every loss is minimized by the same posterior means at every \\(\\x\\) and \\(t\\). A real network has finite capacity and cannot reach the optimum everywhere, so the factor decides where its remaining error goes. Near \\(t = 0\\) the input is almost pure noise, so \\(\\hx_0 \\approx \\x\\) is easy, and the noise loss, with weight \\(t^2/(1-t)^2\\), barely penalizes a poor \\(\\hx_1\\). Near \\(t = 1\\) the same weight becomes very large. The velocity loss gives a weight of at least 1 at every \\(t\\), which grows like \\((1-t)^{-2}\\) near \\(t = 1\\). The lower plot shows the four weights on a log scale. §2.5 of the survey returns to the choice of weighting.'],
    check: {
      q: 'For the linear path at \\(t = 0.5\\), by what factor does the velocity loss weight an error in \\(\\hx_1\\)?',
      answer: 4, tol: 0.01,
      fb: { true: 'Right. With α̇ = 1, σ̇ = −1 the weight is (1 + t/(1 − t))² = (1 − t)^{−2} = 4 at t = 0.5. The lower plot shows the same value at t = 0.5.', false: 'Insert α̇_t = 1, σ̇_t = −1, α_t = σ_t = 0.5 into the bracket: 1 − (−1)(0.5/0.5) = 2, and its square is 4. You can also read the velocity curve at t = 0.5.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>This closes the loop with the first tutorial. Annealed Langevin dynamics needed the score of \\(p_\\sigma\\) at every level of a noise ladder. One network \\(\\s_\\theta(\\x, t)\\), trained with \\(J_{\\mathrm{DSM}}\\) at random times, provides all of them from data alone. With \\(\\sigma = \\sigma_t/\\alpha_t\\) and the rescaling by \\(\\alpha_t\\) undone, it can replace the exact mixture score in the last step of <a href="../ebm-langevin/#annealing">Energies, scores and Langevin dynamics</a>.</p>
      <p><a href="../sde/">§2.3 of the survey</a> makes the noise level continuous and shows that the score learned here drives both a reverse-time SDE and a deterministic probability-flow ODE. §2.4 replaces the score target by the velocity target of the last step. The tutorial <a href="../ddpm/">DDPM: a chain of small Gaussian steps</a> reaches the same denoising loss from a variational bound and shows that its noise predictor is a rescaled score.</p>
      <p>The animation <a href="../../animations.html#a14">A14, posterior sampling</a> plugs Tweedie’s denoiser into a likelihood to solve inverse problems.</p>`,
  },
];
