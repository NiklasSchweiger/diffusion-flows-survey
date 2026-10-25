// Content of the §2.3 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const y = S('y'), tau = S('tau'), rho = S('rho'), b = S('drift'), g2 = `${S('g')}^2`, div = S('div'), lap = S('lap');
const glr = S('glr'), dtau = `\\dd${tau}`, ratio = `\\frac{${S('adot')}}{${S('alpha')}}`;
const sig2 = `${S('sig')}^2`, lamd = S('lamdot'), glp = `${S('glp')}(${S('x')})`;

export const INTRO = `
  <div class="kicker">Survey §2.3 · Interactive tutorial</div>
  <h1>Diffusion as an SDE: Fokker–Planck, time reversal and the probability-flow ODE</h1>
  <p>Annealed Langevin dynamics in the first tutorial used a finite ladder of noise levels. Letting the number of levels go to infinity turns the corruption into a stochastic differential equation. This tutorial derives the three equations behind every continuous-time diffusion model: the Fokker–Planck equation, the reverse-time SDE and the probability-flow ODE. Each step states one idea, shows one formula and gives you a plot that the formula controls. Every symbol can be hovered or tapped, and the button "Symbols" in the top bar opens the full list.</p>
  <p>As in the survey, steps 1 to 4 write the corruption in its natural direction, from data to noise, with time \\(\\tau = 1 - t\\), and step 5 onwards returns to \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), where \\(t = 0\\) is noise and \\(t = 1\\) is data. The path is linear, \\(\\alpha_t = t\\) and \\(\\sigma_t = 1 - t\\), except in step 7. The data are the Gaussian mixtures of the first two tutorials, so every density and score on this page is exact.</p>
  <ol class="toc">
    <li><a href="#forward">The forward SDE</a></li>
    <li><a href="#fp">How the density moves: the Fokker–Planck equation</a></li>
    <li><a href="#transport">Diffusion is transport along the score</a></li>
    <li><a href="#pfode">The probability-flow ODE</a></li>
    <li><a href="#reverse">Running the process backwards</a></li>
    <li><a href="#ours">Two samplers in the survey's convention</a></li>
    <li><a href="#schedules">Schedules are reparametrizations</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'forward', kicker: 'The forward process', title: 'The forward SDE', scene: 'forward',
    prose: [
      'Start from a data point \\(\\y_0\\sim p_{\\mathrm{data}}\\) and let it evolve by a linear SDE. In each short time step it is shrunk towards zero by the drift \\(f_\\tau\\y_\\tau\\,\\dd\\tau\\) and receives a Gaussian kick \\(g_\\tau\\,\\dd\\W_\\tau\\). The plot shows 40 of 1000 such paths. The shading behind them is the exact density \\(\\rho_\\tau\\) of \\(\\y_\\tau\\), and the panel on the right compares a histogram of all 1000 paths at the chosen \\(\\tau\\) with that density.',
      'The coefficients are not free. If the SDE is to produce the same marginals as the interpolation \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) of the survey, \\(f\\) and \\(g\\) are fixed by \\(\\alpha_t\\) and \\(\\sigma_t\\), as in the second line below. Move \\(\\tau\\) and watch the readout: \\(g^2\\) is small near the data and grows without bound near the noise end, where \\(f\\) also diverges. The paths are therefore stopped at \\(\\tau = 0.97\\).',
    ],
    tex: `\\begin{aligned} \\dd${y} &= ${S('f')}\\,${y}\\,${dtau} + ${S('g')}\\,\\dd${S('W')}, \\qquad \\y_0\\sim ${S('pdata')} \\\\ ${S('f')} &= -${ratio}, \\qquad ${g2} = 2${sig2}\\Big(${ratio} - \\frac{${S('sdot')}}{${S('sig')}}\\Big) = ${sig2}\\,${lamd} \\end{aligned}`,
    deriv: {
      have: ['A linear SDE \\(\\dd\\y_\\tau = f_\\tau\\y_\\tau\\,\\dd\\tau + g_\\tau\\,\\dd\\W_\\tau\\) whose coefficients f and g are unknown.', 'The marginals it should have: \\(\\y_\\tau = \\x_{1-\\tau}\\) with \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\).'],
      goal: `${S('f')} = -${ratio}, \\qquad ${g2} = ${sig2}\\,${lamd}`,
      plan: 'Compute the mean and the variance of the linear SDE, match them to \\(\\alpha_t\\) and \\(\\sigma_t^2\\), and solve for f and g.',
      title: 'Derivation of f and g', lines: [
      { tex: `\\E[${y}\\mid\\y_0] = ${S('a')}\\,\\y_0, \\qquad \\dot a_\\tau = ${S('f')}\\,${S('a')}, \\quad a_0 = 1`, why: 'Take the conditional expectation of the SDE. The Brownian increment has mean zero, so the mean obeys the ODE without noise.', uses: ['expectation', 'brownian', 'linear-ode'] },
      { tex: `\\frac{\\dd}{\\dd\\tau}${S('b2')} = 2${S('f')}\\,${S('b2')} + ${CHG(g2)}, \\quad b_0 = 0`, why: 'Apply Itô’s formula to \\((\\y_\\tau - a_\\tau\\y_0)^2\\). The drift doubles the variance rate \\(2fb^2\\), and the noise adds \\(g^2\\) per unit time.', uses: ['ito', 'linear-ode'] },
      { tex: `${S('a')} = ${S('alpha')}\\big|_{t=1-\\tau} \\;\\Rightarrow\\; ${S('f')} = \\frac{\\dot a_\\tau}{a_\\tau} = ${CHG(`-${ratio}`)}`, why: 'Match the mean factor to the signal scale. Since \\(t = 1-\\tau\\), \\(\\dd/\\dd\\tau = -\\dd/\\dd t\\), which gives the minus sign.', uses: ['time-change', 'linear-ode'] },
      { tex: `${S('b2')} = ${sig2}\\big|_{t=1-\\tau} \\;\\Rightarrow\\; ${g2} = -2${S('sig')}${S('sdot')} + 2${ratio}${sig2} = ${CHG(`2${sig2}\\Big(${ratio} - \\frac{${S('sdot')}}{${S('sig')}}\\Big)`)}`, why: 'Match the variance and solve the second line for \\(g^2\\): \\(g^2 = \\dd(b^2)/\\dd\\tau - 2fb^2\\).' },
      { tex: `${S('lam')} = \\log\\frac{${S('alpha')}^2}{${sig2}} \\;\\Rightarrow\\; ${lamd} = 2\\Big(${ratio} - \\frac{${S('sdot')}}{${S('sig')}}\\Big), \\quad ${g2} = ${CHG(`${sig2}\\,${lamd}`)}`, why: 'The bracket is half the time derivative of the log signal-to-noise ratio \\(\\lambda_t\\).', uses: ['log-rules', 'log-deriv'] },
    ] },
    after: ['Every Gaussian interpolation whose log-SNR increases in \\(t\\) is the marginal law of some linear SDE. The variance-preserving process of DDPM has \\(\\alpha_t^2 + \\sigma_t^2 = 1\\), and the variance-exploding process of the first score-based models has \\(\\alpha_t = 1\\). Step 7 shows that the choice between them is a change of coordinates.'],
    check: {
      q: 'For the linear path, what is \\(g^2\\) at \\(t = 0.5\\), that is at \\(\\tau = 0.5\\)?',
      answer: 2, tol: 0.01,
      fb: { true: 'Right. With α_t = t and σ_t = 1 − t the bracket is 1/t + 1/(1 − t) = 4 at t = 0.5, and 2 · 0.25 · 4 = 2. Set τ = 0.5: the readout shows g² = 2.00.', false: 'Insert α̇_t = 1, σ̇_t = −1 and α_t = σ_t = 0.5 into the second line: the bracket is 1/0.5 + 1/0.5 = 4, and 2σ_t² times 4 is 2 · 0.25 · 4.' },
    },
  },
  {
    id: 'fp', kicker: 'Fokker–Planck', title: 'How the density moves: the Fokker–Planck equation', scene: 'fp',
    prose: [
      'Individual paths are random, but their density \\(\\rho_\\tau\\) changes deterministically. For a general SDE \\(\\dd\\y = \\bm b(\\y,\\tau)\\,\\dd\\tau + g_\\tau\\,\\dd\\W\\), the rate of change of the density at a fixed point is the sum of two terms. The first transports mass along the drift. The second spreads it out: the Laplacian is negative at a peak and positive in a dip, so it moves mass from peaks to dips.',
      'The lower panel draws both terms for the forward process of step 1, where \\(\\bm b = f_\\tau\\y\\), and their sum. The dots are the actual rate of change \\(\\partial_\\tau\\rho_\\tau\\), computed from the exact density by a finite difference in \\(\\tau\\). Move \\(\\tau\\): the sum passes through every dot. Early on, the diffusion term dominates and erodes the three peaks. Later, the transport term pulls the mass inwards while the diffusion term widens it.',
    ],
    tex: `${S('dt')} = -${div}\\big(${b}\\,${rho}\\big) + \\tfrac12\\,${g2}\\,${lap}${rho}`,
    deriv: {
      have: ['An SDE \\(\\dd\\y_\\tau = \\bm b(\\y_\\tau)\\,\\dd\\tau + g_\\tau\\,\\dd\\W_\\tau\\) and the density \\(\\rho_\\tau\\) of its state.', 'Itô’s formula, which says how a smooth function of the state changes along the SDE.'],
      goal: 'step',
      plan: 'Follow a smooth test function φ along the SDE, take expectations, and integrate by parts until φ stands alone. Equality for every φ gives an equation for the density.',
      title: 'Derivation in three lines', lines: [
      { tex: `\\dd${S('phi')}(${y}) = \\Big(${b}\\cdot\\nabla${S('phi')} + \\tfrac12${g2}\\,${lap}${S('phi')}\\Big)\\dd${tau} + \\text{martingale}`, why: 'Itô’s formula for a smooth test function φ. The second-order term appears because \\(\\dd\\W^2 = \\dd\\tau\\).', uses: ['ito', 'brownian'] },
      { tex: `\\frac{\\dd}{\\dd${tau}}\\int ${S('phi')}\\,${rho}\\,\\dd\\y = \\int \\Big(${b}\\cdot\\nabla${S('phi')} + \\tfrac12${g2}\\,${lap}${S('phi')}\\Big)${CHG(rho)}\\,\\dd\\y`, why: 'Take expectations. The martingale has mean zero, and an expectation is an integral against \\(\\rho_\\tau\\).', uses: ['expectation', 'brownian'] },
      { tex: `\\int ${S('phi')}\\,${S('dt')}\\,\\dd\\y = \\int ${S('phi')}\\,\\Big(${CHG(`-${div}(${b}\\,${rho}) + \\tfrac12${g2}\\,${lap}${rho}`)}\\Big)\\dd\\y`, why: 'Integrate by parts, once in the drift term and twice in the Laplacian term, to move the derivatives from φ to \\(\\rho_\\tau\\).', uses: ['ibp'] },
      { tex: `${S('dt')} = -${div}\\big(${b}\\,${rho}\\big) + \\tfrac12\\,${g2}\\,${lap}${rho}`, why: 'The two sides agree for every test function φ, so the integrands agree.', uses: ['test-functions'] },
    ] },
    after: ['The Langevin equation of the first tutorial is the special case \\(\\bm b = \\nabla\\log p\\), \\(g = \\sqrt2\\). There, the two terms cancel exactly when \\(\\rho = p\\), which is why \\(p\\) is stationary.'],
    check: {
      q: 'At the top of a peak of \\(\\rho_\\tau\\), what sign does the diffusion term \\(\\tfrac12 g^2\\Delta\\rho_\\tau\\) have?',
      opts: [['Positive: mass flows in', false], ['Negative: mass flows out', true], ['Zero, because the slope is zero', false]],
      fb: { true: 'Right. At a maximum the second derivative is negative, so the diffusion term lowers the peak. In the lower panel the diffusion curve dips below zero under each peak.', false: 'The first derivative is zero at a peak, but the Laplacian is the second derivative, which is negative at a maximum. Look at the diffusion curve in the lower panel under each peak.' },
    },
  },
  {
    id: 'transport', kicker: 'One equation, two readings', title: 'Diffusion is transport along the score', scene: 'transport',
    prose: [
      'The key step of the section is to write the diffusion term as a transport term. Because \\(\\nabla\\rho = \\rho\\nabla\\log\\rho\\), the Laplacian is the divergence of \\(\\rho\\nabla\\log\\rho\\), and Fokker–Planck becomes a continuity equation. Spreading by noise has the same effect on the density as moving every point with the velocity \\(-\\tfrac12 g^2\\nabla\\log\\rho_\\tau\\), which points away from the peaks.',
      'The plot draws the drift \\(\\bm b = f_\\tau\\y\\), the diffusion velocity, and their sum, the bold curve. The arrows under the density show where each point moves. The lower panel checks the claim: the continuity form \\(-\\nabla\\cdot(\\text{velocity}\\cdot\\rho_\\tau)\\) passes through the same dots as in step 2. Hover over the plot to read off the three velocities at one point.',
    ],
    tex: `\\begin{aligned} ${lap}${rho} &= ${div}\\big(${rho}\\,${glr}\\big) \\\\ ${S('dt')} &= -${div}\\Big(\\big(${S('vel')}\\big)\\,${rho}\\Big) \\end{aligned}`,
    deriv: {
      have: ['The Fokker–Planck equation of step 2, with a transport term and a diffusion term.'],
      goal: 'step',
      plan: 'Write the Laplacian as the divergence of \\(\\nabla\\rho_\\tau = \\rho_\\tau\\nabla\\log\\rho_\\tau\\). Then both terms are divergences of a velocity times \\(\\rho_\\tau\\), and they combine into one.',
      lines: [
      { tex: `\\nabla${rho} = ${rho}\\,${CHG(`\\frac{\\nabla${rho}}{${rho}}`)} = ${rho}\\,${glr}`, why: 'Multiply and divide by \\(\\rho_\\tau\\), and use the chain rule \\(\\nabla\\log\\rho = \\nabla\\rho/\\rho\\).', uses: ['log-deriv'] },
      { tex: `${lap}${rho} = ${div}\\nabla${rho} = ${CHG(`${div}\\big(${rho}\\,${glr}\\big)`)}`, why: 'The Laplacian is the divergence of the gradient.', uses: ['div-grad-lap'] },
      { tex: `${S('dt')} = -${div}(${b}\\,${rho}) + ${div}\\big(\\tfrac12${g2}\\,${rho}\\,${glr}\\big) = ${CHG(`-${div}\\Big(\\big(${S('vel')}\\big)\\,${rho}\\Big)`)}`, why: 'Insert into Fokker–Planck. \\(g^2\\) depends only on τ, so it moves inside the divergence, and both terms combine.', uses: ['fokker-planck', 'grad-linear'] },
    ] },
    after: ['The result is a continuity equation, the equation of a density that is carried along by a velocity field without any noise. Step 4 reads it forwards, step 5 backwards.'],
    check: {
      q: 'Just to the left of the right-most peak of \\(\\rho_\\tau\\), which way does the diffusion velocity \\(-\\tfrac12 g^2\\nabla\\log\\rho_\\tau\\) point?',
      opts: [['Right, towards the peak', false], ['Left, away from the peak', true], ['It is zero there', false]],
      fb: { true: 'Right. Left of a peak the density increases, so ∇log ρ is positive and the minus sign makes the velocity negative. Diffusion moves mass away from the peaks, which is how the density spreads out.', false: 'Left of a peak the density increases towards the peak, so the score ∇log ρ is positive. The diffusion velocity has a minus sign, so it points left, away from the peak. Hover there and read the readout.' },
    },
  },
  {
    id: 'pfode', kicker: 'Reading forwards', title: 'The probability-flow ODE', scene: 'pfode',
    prose: [
      'Read the continuity equation of step 3 forwards. If every point moves deterministically with the velocity \\(\\bm b - \\tfrac12 g^2\\nabla\\log\\rho_\\tau\\), the density of the moving points obeys the same equation as the density of the SDE. Both start from \\(p_{\\mathrm{data}}\\), so they agree at every \\(\\tau\\). The randomness of the SDE has been traded for a dependence of the velocity on the current density, through its score.',
      'The grey paths are the SDE paths of step 1. The coloured paths solve the ODE, each started at a quantile of the data, from 2% to 98%. They are smooth, they never cross, and they keep their quantile: the path that starts at the 10% quantile of the data stays at the 10% quantile of \\(\\rho_\\tau\\). Move \\(\\tau\\) and compare the coloured ticks with the histogram on the right.',
    ],
    tex: `\\frac{\\dd${y}}{\\dd${tau}} = ${b}(${y}) - \\tfrac12\\,${g2}\\,${glr}(${y})`,
    after: ['Individual paths of the two processes are entirely different, only their densities agree. The ODE gives a deterministic, invertible map from data to noise, which is what DDIM sampling, exact likelihoods and the flow models of §2.4 use.'],
    check: {
      q: 'Can two paths of the probability-flow ODE cross in one dimension?',
      opts: [['Yes, wherever the density has two modes', false], ['No', true], ['Only near the noise end', false]],
      fb: { true: 'Right. Through each point at each time there is exactly one solution of the ODE, so two paths that met would coincide from then on. In 1D this means the order of the points is preserved, and so is each quantile.', false: 'An ODE with a smooth velocity has exactly one solution through each point at each time. Two paths that met would coincide from then on. Look at the coloured paths: they never cross.' },
    },
  },
  {
    id: 'reverse', kicker: 'Reading backwards', title: 'Running the process backwards', scene: 'reverse',
    prose: [
      'To generate data we need the reverse direction, from noise at \\(t = 0\\) to data at \\(t = 1\\). Write \\(p_t = \\rho_{1-t}\\). Reversing time flips the sign of the continuity equation, and adding and subtracting a diffusion term turns it into a Fokker–Planck equation with a new drift. The result is the reverse-time SDE of Anderson (1982), the first line below.',
      'The same argument works with any amount of added diffusion, which gives the family in the second line. At \\(\\eta = 1\\) it is the reverse SDE; at \\(\\eta = 0\\) it is the probability-flow ODE of step 4, run backwards. The plot runs 1000 particles from the exact \\(p_{0.03}\\) to \\(t = 1\\) with fine steps and the exact score. Move \\(\\eta\\): the paths change completely, but the histogram at every \\(t\\) stays on the exact density, up to the noise of 1000 samples.',
    ],
    tex: `\\begin{aligned} \\dd${S('x')} &= \\big(-${b} + ${g2}\\,${glp}\\big)\\,\\dd${S('t')} + ${S('g')}\\,\\dd${S('W')} \\\\ \\dd${S('x')} &= \\Big(-${b} + \\tfrac{1+${S('eta')}^2}{2}\\,${g2}\\,${glp}\\Big)\\dd${S('t')} + ${S('eta')}\\,${S('g')}\\,\\dd${S('W')} \\end{aligned}`,
    deriv: {
      have: ['The forward Fokker–Planck equation, written as transport along \\(\\bm b - \\tfrac12 g^2\\nabla\\log\\rho_\\tau\\) in step 3.', 'The reversed density \\(p_t = \\rho_{1-t}\\), which runs from noise to data.'],
      goal: 'step',
      plan: 'Flip the sign of time in the transport form. Then add and subtract a diffusion term, so that the equation is again a Fokker–Planck equation, now of an SDE that runs from noise to data.',
      lines: [
      { tex: `\\partial_{${S('t')}}${S('pt')} = -\\partial_\\tau\\rho_\\tau = ${CHG('+')}${div}\\Big(\\big(${b} - \\tfrac12${g2}\\nabla\\log ${S('pt')}\\big)${S('pt')}\\Big)`, why: '\\(p_t = \\rho_{1-t}\\), so \\(\\partial_t = -\\partial_\\tau\\). Insert the continuity form of step 3.', uses: ['time-change', 'transport'] },
      { tex: `\\partial_{${S('t')}}${S('pt')} = -${div}\\Big(\\big(-${b} + \\tfrac12${g2}\\nabla\\log ${S('pt')}\\big)${S('pt')}\\Big)`, why: 'Move the sign inside: the reversed continuity equation has velocity \\(-\\bm b + \\tfrac12 g^2\\nabla\\log p_t\\).' },
      { tex: `= -${div}\\Big(\\big(-${b} + ${CHG(`${g2}`)}\\nabla\\log ${S('pt')}\\big)${S('pt')}\\Big) + ${CHG(`\\tfrac12${g2}\\,${lap}${S('pt')}`)}`, why: 'Add and subtract \\(\\tfrac12 g^2\\Delta p_t = \\diver(\\tfrac12 g^2 p_t\\nabla\\log p_t)\\), the identity of step 3. The equation is unchanged.', uses: ['transport'] },
      { tex: '\\Rightarrow\\;\\text{Fokker–Planck with drift } -\\bm b + g^2\\nabla\\log p_t \\text{ and diffusion } g', why: 'Compare with step 2. An SDE with this drift and diffusion has the marginals \\(p_t\\), which gives the first line of the formula.', uses: ['fokker-planck'] },
      { tex: `\\text{add and subtract } \\tfrac12${S('eta')}^2${g2}\\,${lap}${S('pt')} \\;\\Rightarrow\\; \\text{drift } -${b} + ${CHG(`\\tfrac{1+${S('eta')}^2}{2}`)}${g2}\\nabla\\log p_t,\\ \\text{diffusion } ${S('eta')}${S('g')}`, why: 'Doing the same with any fraction \\(\\eta^2\\) of the diffusion gives the second line. \\(\\eta = 0\\) adds nothing and leaves the reversed ODE.', uses: ['transport'] },
    ] },
    after: ['Anderson proved more: the reversed process has the law of the forward process as a whole path, not only at each time. The derivation above gives the statement about marginals, which is all that sampling needs.'],
    check: {
      q: 'Set \\(\\eta = 0\\). Which equation does the sampler integrate?',
      opts: [['The forward SDE of step 1, backwards', false], ['The probability-flow ODE of step 4, backwards', true], ['Langevin dynamics at each t', false]],
      fb: { true: 'Right. At η = 0 the noise vanishes and the drift is −b + ½g²∇log p_t, the negative of the probability-flow velocity. The paths are the coloured paths of step 4, traversed from noise to data.', false: 'At η = 0 there is no noise, and the drift is −(b − ½g²∇log p_t): the probability-flow velocity of step 4 with the opposite sign, because time runs the other way.' },
    },
  },
  {
    id: 'ours', kicker: 'Back to our convention', title: 'Two samplers in the survey’s convention', scene: 'ours',
    prose: [
      'Substituting \\(\\bm b = -(\\dot\\alpha_t/\\alpha_t)\\x\\) and \\(g^2 = \\sigma_t^2\\dot\\lambda_t\\) from step 1 writes both samplers in terms of the interpolation. A learned score \\(\\s_\\theta\\approx\\nabla\\log p_t\\) turns either equation into a generative model. DDPM ancestral sampling is a discretization of the second equation and DDIM a discretization of the first.',
      'The two panels run both equations on the 2D mixture of §2.2, from 500 draws of \\(\\N(0, I)\\) at \\(t = 0.02\\) to \\(t = 0.995\\), with the exact score and \\(K\\) plain Euler or Euler–Maruyama steps. The steps are uniform in \\(\\lambda_t\\). Press "Run" to watch, and lower \\(K\\): with few steps the SDE overshoots and scatters its samples, while the ODE stays close to the data.',
    ],
    tex: `\\begin{aligned} \\dot{${S('x')}} &= ${ratio}\\,${S('x')} + \\tfrac12\\,${sig2}\\,${lamd}\\,${glp} \\\\ \\dd${S('x')} &= \\Big(${ratio}\\,${S('x')} + ${sig2}\\,${lamd}\\,${glp}\\Big)\\dd${S('t')} + ${S('sig')}\\sqrt{${lamd}}\\,\\dd${S('W')} \\end{aligned}`,
    deriv: {
      have: ['The forward drift \\(\\bm b = f\\y\\) and \\(g^2 = \\sigma_t^2\\dot\\lambda_t\\) of step 1.', 'The probability-flow ODE of step 4 and the reverse SDE of step 5, both written in τ.'],
      goal: 'step',
      plan: 'Insert f and g into both samplers and change the time variable from τ to t, using \\(\\dd\\tau = -\\dd t\\).',
      lines: [
      { tex: `${b} = ${S('f')}\\,\\y = ${CHG(`-${ratio}\\,${S('x')}`)}, \\qquad ${g2} = ${CHG(`${sig2}\\,${lamd}`)}`, why: 'The drift and diffusion coefficient of step 1, with \\(\\y_\\tau = \\x_t\\).', uses: ['forward-sde'] },
      { tex: `\\dot{${S('x')}} = -\\frac{\\dd${y}}{\\dd${tau}} = -\\big(${b} - \\tfrac12${g2}\\,${glp}\\big) = ${CHG(`${ratio}\\,${S('x')} + \\tfrac12\\,${sig2}\\,${lamd}\\,${glp}`)}`, why: 'The probability-flow ODE of step 4 with \\(\\dd\\tau = -\\dd t\\).', uses: ['pf-ode', 'time-change'] },
      { tex: `-${b} + ${g2}\\,${glp} = ${CHG(`${ratio}\\,${S('x')} + ${sig2}\\,${lamd}\\,${glp}`)}, \\qquad ${S('g')} = ${CHG(`${S('sig')}\\sqrt{${lamd}}`)}`, why: 'The drift and diffusion of the reverse SDE of step 5.', uses: ['reverse-sde'] },
    ] },
    after: ['The first equation is the flow-matching velocity of §2.2, step 7, written with the score: \\(\\hat{\\bm u}_t = (\\dot\\alpha_t/\\alpha_t)\\x + \\tfrac12\\sigma_t^2\\dot\\lambda_t\\nabla\\log p_t\\). The SDE has twice the score term, and its noise has to be removed again by that term in the following steps. This is one reason why stochastic samplers need more steps than deterministic ones for the same accuracy.'],
    check: {
      q: 'In the plot, set \\(K = 10\\). Which sampler puts its samples closer to the data?',
      opts: [['The ODE', true], ['The SDE', false], ['Both are equally close', false]],
      fb: { true: 'Right. The readout shows a mean log-density close to that of exact samples for the ODE and a lower one for the SDE, whose samples are spread too widely. At K = 200 both are accurate.', false: 'Compare the mean log-density in the readout with the value for exact samples. At K = 10 the SDE samples are spread too widely, and its mean log-density is lower. At K = 200 both agree.' },
    },
  },
  {
    id: 'schedules', kicker: 'Schedules', title: 'Schedules are reparametrizations', scene: 'schedules',
    prose: [
      'Divide the interpolation by \\(\\alpha_t\\). In the coordinates \\(\\y = \\x_t/\\alpha_t\\) every Gaussian path is data plus noise of standard deviation \\(\\sigma = \\sigma_t/\\alpha_t\\), the variance-exploding path of EDM. Written in \\(\\sigma\\), the probability-flow ODE of step 6 no longer contains \\(\\alpha_t\\) or \\(\\sigma_t\\) at all.',
      'Pick a schedule: the linear path, the variance-preserving cosine path or a variance-exploding path, all over the same range of \\(\\sigma\\). The upper panel shows the probability-flow paths of the 1D mixture as functions of \\(t\\); they look different for each schedule. The lower panel shows the same paths in \\(\\y\\) against \\(\\sigma\\). They are identical for every schedule. What does change is where \\(K\\) equal steps in \\(t\\) land on the \\(\\sigma\\) axis, marked by the dots.',
    ],
    tex: `\\begin{aligned} \\frac{${S('x')}}{${S('alpha')}} &= \\x_1 + \\frac{${S('sig')}}{${S('alpha')}}\\,\\x_0, \\qquad ${S('r')} = \\frac{${S('sig')}}{${S('alpha')}} = e^{-${S('lam')}/2} \\\\ \\frac{\\dd${S('yr')}}{\\dd${S('r')}} &= -${S('r')}\\,\\nabla_{${S('yr')}}\\log ${S('psig')} \\end{aligned}`,
    after: ['Two schedules with the same range of \\(\\lambda\\) define the same family of distributions and the same ODE paths. What changes is how the integration steps are spaced and how training effort is distributed over noise levels, which §2.5 of the survey discusses.'],
    check: {
      q: 'With \\(K = 12\\) equal steps in \\(t\\), which schedule spends the most steps at small noise, \\(\\sigma < 0.3\\)?',
      opts: [['Linear', false], ['Cosine', false], ['Variance exploding', true]],
      fb: { true: 'Right. Its σ falls geometrically in t, so equal steps in t are equal steps in log σ, and a large share lands at small σ. The readout counts the steps below 0.3 for each schedule.', false: 'Switch between the three schedules and read the count of steps below σ = 0.3 in the readout. The variance-exploding path spaces its steps evenly in log σ, so many land at small σ.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>§2.4 of the survey starts from the continuity equation of step 3 and learns the velocity directly, without an SDE: flow matching. The tutorial <a href="../flow-matching/">Flows and flow matching</a> follows that path. The advanced part of §2.3 bounds the error of a discretized reverse SDE by the score error, the step error and the error of starting at \\(\\N(0, I)\\) instead of the exact \\(p_0\\).</p>
      <p>The earlier tutorials, <a href="../ebm-langevin/">Energies, scores and Langevin dynamics</a> and <a href="../score-matching/">Learning the score</a>, show how a score is used and how it is learned. The tutorial <a href="../ddpm/">DDPM</a> is the discrete-time chain whose limit this tutorial takes. The animation <a href="../../animations/flow-matching/">A1, flow matching</a> shows the deterministic transport of step 4 in 2D.</p>`,
  },
];
