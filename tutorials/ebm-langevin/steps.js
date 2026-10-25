// Content of the §2.1 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const grad = (f) => `${S('nabla', '\\nabla')}${f}`;   // ∇ without the subscript, as in the paper's later equations

export const INTRO = `
  <div class="kicker">Survey §2.1 · Interactive tutorial</div>
  <h1>Energies, scores and Langevin dynamics</h1>
  <p>This page works through the first section of the survey one formula at a time. Each step states one idea, shows one formula and gives you a plot that the formula controls. Every symbol can be hovered or tapped to see what it is, what shape it has and where it appears in the plot. The button "Symbols" in the top bar opens the full list.</p>
  <p>All densities and scores on this page are exact. The targets are Gaussian mixtures, so nothing is learned and every difference you see comes from the sampler.</p>
  <ol class="toc">
    <li><a href="#energy">Energy and density</a></li>
    <li><a href="#score">The score does not depend on Z</a></li>
    <li><a href="#langevin">Langevin dynamics</a></li>
    <li><a href="#stationary">Why p is the stationary density</a></li>
    <li><a href="#ula">The unadjusted Langevin algorithm</a></li>
    <li><a href="#smoothing">Why one noise level is not enough</a></li>
    <li><a href="#annealing">Annealed Langevin dynamics</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'energy', kicker: 'Energy-based models', title: 'Energy and density', scene: 'energy',
    prose: [
      'Any positive density can be written through an energy \\(E\\). The exponential makes the density positive, and dividing by the normalizing constant \\(Z\\) makes it integrate to one. Low energy means high probability.',
      'Move the slider \\(c\\), which adds a constant to the energy. The curve \\(e^{-E}\\) and its area \\(Z\\) shrink or grow by the same factor, so their ratio \\(p\\) stays where it is. Computing \\(Z\\) needs an integral over the whole space, yet it carries no information about where the mass is.',
    ],
    tex: `${S('p')}(${S('x')}) = \\frac{e^{-${S('E')}(${S('x')})}}{${S('Z')}}, \\qquad ${S('Z')} = \\int e^{-${S('E')}(${S('x')})}\\,\\dd ${S('x')}`,
    after: ['When the energy is a neural network \\(E_\\theta\\), the integral \\(Z_\\theta\\) has no closed form. Maximum likelihood needs it, and so does any sampler that evaluates \\(p\\) directly. The next step shows a quantity that does not need it.'],
    check: {
      q: 'You replace E by E + 2. What happens to Z?',
      opts: [['It is multiplied by exp(−2)', true], ['It grows by 2', false], ['It does not change', false]],
      fb: { true: 'Right. The constant comes out of the integral: ∫exp(−E−2) dx = exp(−2) ∫exp(−E) dx. Numerator and denominator of p shrink by the same factor, so p is unchanged.', false: 'Not quite. Pull the constant out of the integral: ∫exp(−(E+2)) dx = exp(−2) ∫exp(−E) dx. Set c = 2 on the slider and compare the Z readout with the one at c = 0.' },
    },
  },
  {
    id: 'score', kicker: 'The score', title: 'The score does not depend on Z', scene: 'score',
    prose: [
      'The score is the gradient of the log-density with respect to the data. Move the pointer over the plot to move the probe. The bold arrow is the score at the probe. It points toward higher probability and is long where the density changes quickly, which is on the flanks of the modes and not at their peaks.',
      'Move \\(c\\) again. The energy and \\(Z\\) in the readout change, but no arrow moves, because \\(Z\\) is a constant and its gradient is zero. Step through the derivation below to see where \\(Z\\) drops out.',
    ],
    tex: `${S('s')}(${S('x')}) = ${S('nabla')}\\log ${S('p')}(${S('x')}) = -${S('nabla')} ${S('E')}(${S('x')})`,
    after: ['This is the starting point of the whole chapter. A model that learns \\(\\s\\) instead of \\(p\\) never has to compute \\(Z\\). The rest of this page shows that the score is enough to sample, and §2.2 of the survey shows that it can be learned from samples.'],
    deriv: {
      have: ['The density \\(p(\\x) = e^{-E(\\x)}/Z\\) of step 1. The constant Z does not depend on \\(\\x\\).', 'The definition of the score, \\(\\s(\\x) = \\nabla\\log p(\\x)\\).'],
      goal: `${S('s')}(${S('x')}) = -${S('nabla')} ${S('E')}(${S('x')})`,
      plan: 'Insert the energy form into the definition and split the logarithm. Z then contributes a constant, and the gradient of a constant is zero.',
      lines: [
      { tex: `${S('s')}(${S('x')}) = ${S('nabla')}\\log ${S('p')}(${S('x')})`, why: 'Definition of the score.' },
      { tex: `= ${S('nabla')}\\log ${CHG(`\\frac{e^{-${S('E')}(${S('x')})}}{${S('Z')}}`)}`, why: 'Insert the energy form of p from step 1.' },
      { tex: `= ${S('nabla')}\\big(${CHG(`-${S('E')}(${S('x')})`)}\\big) ${CHG(`- ${S('nabla')}\\log ${S('Z')}`)}`, why: 'The log of a quotient is a difference, and \\(\\log e^{-E} = -E\\).', uses: ['log-rules'] },
      { tex: `= -${S('nabla')} ${S('E')}(${S('x')}) ${CHG('+\\,0')}`, why: 'Z does not depend on x, so its gradient is zero.', uses: ['grad-linear'] },
    ] },
  },
  {
    id: 'langevin', kicker: 'Langevin dynamics', title: 'Langevin dynamics', scene: 'langevin',
    prose: [
      'The overdamped Langevin equation moves a particle with two terms. The drift \\(\\nabla\\log p\\,\\dd\\tau\\) pushes it uphill along the score. The noise \\(\\sqrt2\\,\\dd\\W_\\tau\\) adds a random kick of variance \\(2\\,\\dd\\tau\\) per coordinate. The time \\(\\tau\\) belongs to the sampler and has nothing to do with the interpolation time \\(t\\) of later chapters.',
      'Switch the two terms on and off. With the drift alone, every particle climbs to the nearest peak and stays there. With the noise alone, the cloud spreads without limit. Only with both terms does the cloud settle into the shape of \\(p\\).',
    ],
    tex: `\\dd ${S('xtau')} = ${grad('\\log ' + S('p') + '(' + S('xtau') + ')')}\\,${S('dtau')} + \\sqrt{2}\\,${S('dW')}`,
    after: ['The plot simulates the equation with a small fixed step \\(\\Delta\\tau = 0.01\\). That discretization is the algorithm of step 5 with \\(\\eta = 0.01\\).'],
    check: {
      q: 'Which term spreads mass?',
      opts: [['The drift ∇log p dτ', false], ['The noise √2 dW', true], ['Both terms equally', false]],
      fb: { true: 'Right. The drift concentrates mass at the peaks and the noise spreads it out. The factor √2 is what makes the balance land exactly on p, as the next step shows.', false: 'Switch the noise off and watch the spread readout. The drift moves particles toward the peaks, which reduces the spread. Only the noise increases it.' },
    },
  },
  {
    id: 'stationary', kicker: 'Fokker–Planck equation', title: 'Why p is the stationary density', scene: 'stationary',
    prose: [
      'Follow many particles instead of one. Their density \\(\\rho_\\tau\\) obeys the Fokker–Planck equation on the left of the formula: the drift transports mass along the score, and the noise term \\(\\Delta\\rho_\\tau\\) flattens peaks of \\(\\rho_\\tau\\). The right-hand side is the same equation after one rewrite, which you can step through below.',
      'In the rewritten form the flux is \\(\\rho_\\tau\\nabla\\log(\\rho_\\tau/p)\\). When \\(\\rho_\\tau = p\\) the ratio is one everywhere, its log is zero, and nothing moves. Drag \\(\\tau\\) or press Play to see 3000 particles start in a narrow bump to the right of all three modes and spread into \\(p\\), while the KL divergence to \\(p\\) falls.',
    ],
    tex: `${S('partial')}${S('rho')} = -${S('div')}\\big(${S('rho')}\\,${grad('\\log ' + S('p'))}\\big) + ${S('lap')}${S('rho')} = ${S('div')}\\Big(${S('rho')}\\,${grad('\\log')}\\frac{${S('rho')}}{${S('p')}}\\Big)`,
    deriv: {
      have: ['The Langevin SDE \\(\\dd\\x_\\tau = \\nabla\\log p(\\x_\\tau)\\,\\dd\\tau + \\sqrt2\\,\\dd\\W_\\tau\\) of step 3, and the density \\(\\rho_\\tau\\) of its state at time τ.', 'The Fokker–Planck equation, which says how the density of any SDE evolves.'],
      goal: 'step',
      plan: 'Write down Fokker–Planck for this SDE and rewrite the Laplacian as a divergence, so that both terms combine into one. Then differentiate the KL divergence along the flow to see that it can only decrease.',
      lines: [
      { tex: `${S('partial')}${S('rho')} = -${S('div')}\\big(${S('rho')}\\,${grad('\\log ' + S('p'))}\\big) + ${S('lap')}${S('rho')}`, why: 'Fokker–Planck equation of the Langevin SDE, derived in §2.3 of the survey. The drift gives the transport term and the noise \\(\\sqrt2\\,\\dd\\W\\) gives \\(\\Delta\\rho\\).', uses: ['fokker-planck'] },
      { tex: `${S('lap')}${S('rho')} = ${S('div')}\\big(${CHG(grad(S('rho')))}\\big) = ${S('div')}\\big(${CHG(`${S('rho')}\\,${grad('\\log ' + S('rho'))}`)}\\big)`, why: '\\(\\Delta = \\diver\\nabla\\), and \\(\\nabla\\log\\rho = \\nabla\\rho/\\rho\\), so \\(\\nabla\\rho = \\rho\\nabla\\log\\rho\\).', uses: ['div-grad-lap', 'log-deriv'] },
      { tex: `${S('partial')}${S('rho')} = ${S('div')}\\big(${CHG(`${S('rho')}\\,${grad('\\log ' + S('rho'))} - ${S('rho')}\\,${grad('\\log ' + S('p'))}`)}\\big)`, why: 'Insert the second line into the first and collect both terms under one divergence.', uses: ['grad-linear'] },
      { tex: `= ${S('div')}\\Big(${S('rho')}\\,${grad(CHG(`\\log\\frac{${S('rho')}}{${S('p')}}`))}\\Big)`, why: '\\(\\log\\rho - \\log p = \\log(\\rho/p)\\). If \\(\\rho = p\\) this is zero everywhere, so \\(\\partial\\rho/\\partial\\tau = 0\\) and p is stationary.', uses: ['log-rules'] },
      { tex: `\\frac{\\dd}{\\dd${S('tau')}}${S('KL')}(${S('rho')}\\,\\|\\,${S('p')}) = ${CHG(`-\\int ${S('rho')}\\,\\Big\\|${grad('\\log')}\\frac{${S('rho')}}{${S('p')}}\\Big\\|^2 \\dd${S('x')}`)} \\le 0`, why: 'Differentiate \\(\\KL = \\int\\rho\\log(\\rho/p)\\), insert the previous line and integrate by parts. KL decreases until \\(\\rho = p\\).', uses: ['kl', 'diff-under-int', 'ibp'] },
    ] },
    after: ['The last line says more than stationarity. \\(\\KL(\\rho_\\tau\\,\\|\\,p)\\) decreases along the dynamics, and Jordan, Kinderlehrer and Otto (1998) showed that the equation is exactly the gradient flow of this KL divergence when distributions are compared with the Wasserstein-2 distance. Langevin dynamics is steepest descent on the KL divergence to the target, and it only needs the score.'],
  },
  {
    id: 'ula', kicker: 'Discretization', title: 'The unadjusted Langevin algorithm', scene: 'ula',
    prose: [
      'A computer cannot follow the SDE exactly. The Euler–Maruyama discretization with step \\(\\eta\\) replaces \\(\\dd\\tau\\) by \\(\\eta\\) and \\(\\sqrt2\\,\\dd\\W_\\tau\\) by a Gaussian kick of variance \\(2\\eta\\). Press "One step" to see one update of the tracked particle split into its drift arrow and its kick arrow.',
      'The algorithm is called unadjusted because nothing corrects the discretization error. Its stationary density is not exactly \\(p\\), and the error grows with \\(\\eta\\). The lower plot measures this on the simplest target, a standard Gaussian. Its stationary variance is \\(1/(1-\\eta/2)\\) instead of 1, and the iteration diverges once \\(\\eta \\ge 2\\).',
    ],
    tex: `${S('xk1')} = ${S('xk')} + ${S('eta')}\\, ${S('s')}(${S('xk')}) + \\sqrt{2${S('eta')}}\\; ${S('xi')}, \\qquad ${S('xi')} \\sim \\N(0, I)`,
    after: ['On the mixture in the upper plot, the narrowest scale is the component variance 0.25, and ULA becomes unstable there once η exceeds 2 × 0.25 = 0.5. Step sizes in practice are set by the sharpest feature of the target, not by its overall size.'],
    check: {
      q: 'For the target N(0, 1), the score is s(x) = −x. At η = 1, what variance does ULA settle at?',
      answer: 2, tol: 0.05,
      fb: {
        true: 'Right. With s(x) = −x the update is x′ = (1 − η) x + √(2η) ξ. A stationary variance v satisfies v = (1 − η)² v + 2η, so v = 1/(1 − η/2), which is 2 at η = 1. Set η = 1 and the lower plot settles at 2.',
        false: 'Not quite. Insert s(x) = −x: x′ = (1 − η) x + √(2η) ξ. Take the variance of both sides and set it equal: v = (1 − η)² v + 2η. Solve for v and put η = 1. You can also set η = 1 on the slider and read the lower plot.',
      },
    },
  },
  {
    id: 'smoothing', kicker: 'Noise-smoothed targets', title: 'Why one noise level is not enough', scene: 'smoothing',
    prose: [
      'The target now has three narrow, well separated modes with weights 0.55, 0.30 and 0.15. Between the modes the density is almost zero, so the score there is almost the only force, and it points toward whichever mode is closer. A particle that starts near the small mode stays there. The time to cross a low-density barrier grows exponentially with its height, so ULA keeps the split of its starting positions instead of the target weights.',
      'Smoothing the target with Gaussian noise of scale \\(\\sigma\\) fills in the gaps. Move \\(\\sigma\\) and watch the bars. At small \\(\\sigma\\) the samples have the right shape but the wrong weights. At large \\(\\sigma\\) particles move between modes freely, but they sample the blurred density \\(p_\\sigma\\), not \\(p\\).',
    ],
    tex: `${S('psig')} = ${S('p')} ${S('conv')} \\N(0, ${S('sigma')}^2 I), \\qquad ${S('ssig')}(${S('x')}) = ${S('nabla')}\\log ${S('psig')}(${S('x')})`,
    after: ['Every run uses 300 particles from the same uniform start, ULA with step η = γσ² (γ = 0.15) and the exact score of \\(p_\\sigma\\). For a Gaussian mixture, \\(p_\\sigma\\) is again a mixture, with each component variance increased by \\(\\sigma^2\\).'],
    check: {
      q: 'At σ = 0.25 the score is exact, yet the mass per mode comes out wrong. Why?',
      opts: [
        ['The score is inaccurate between the modes', false],
        ['Particles cannot cross the low-density gaps in the available steps, so they keep the split of their start', true],
        ['The step size η is too large', false],
      ],
      fb: {
        true: 'Right. The sampler is correct in the limit of infinitely many steps, but the expected time to cross a gap is far longer than any run. The weights are frozen by the starting positions.',
        false: 'The score here is exact and η = γσ² is well inside the stable range. Look at where the particles of each mode came from: none of them crosses a gap. The problem is mixing time, not accuracy.',
      },
    },
  },
  {
    id: 'annealing', kicker: 'Annealing', title: 'Annealed Langevin dynamics', scene: 'annealing',
    prose: [
      'Annealing combines the two regimes of step 6. It runs ULA for \\(K = 30\\) steps at each of \\(L\\) noise levels from \\(\\sigma_1 = 3\\) down to \\(\\sigma_L = 0.25\\). At the large levels the modes overlap, particles move between them, and the weights are set. At the small levels each particle is refined inside its mode.',
      'The left panel runs the same number of steps at the smallest level only. Both runs start from the same positions and use the same random numbers. Increase \\(L\\) and compare the bars with the target weights.',
    ],
    tex: `\\begin{aligned} ${S('x')}_{k+1} &= ${S('x')}_k + \\eta_{${S('ell')}}\\, \\s_{${S('sigma')}_{${S('ell')}}}(${S('x')}_k) + \\sqrt{2\\eta_{${S('ell')}}}\\; ${S('xi')}, \\qquad \\eta_{${S('ell')}} = ${S('gamma')}\\,${S('sigma')}_{${S('ell')}}^2 \\\\ ${S('sigma')}_1 &> ${S('sigma')}_2 > \\dots > ${S('sigma')}_{${S('L')}}, \\qquad ${S('ell')} = 1, \\dots, ${S('L')} \\end{aligned}`,
    check: {
      q: 'Both runs end at the same \\(\\sigma_L\\) with the same number of steps and an exact score. Why does only the annealed run get the weights right?',
      opts: [
        ['Particles cross between modes at the large σ levels, where the gaps are filled in', true],
        ['The annealed run uses more steps', false],
        ['The annealed run has a more accurate score', false],
      ],
      fb: {
        true: 'Right. The weights are decided early, at the levels where \\(p_\\sigma\\) has no barriers. The small levels only sharpen each mode, which needs few steps because no particle has to cross a gap.',
        false: 'Both runs use L × 30 steps and both use the exact score of \\(p_\\sigma\\) at their level. The difference is where the particles are allowed to cross between modes.',
      },
    },
    closing: `<h3>Where this leads</h3>
      <p>Making the sequence of noise levels continuous turns annealed Langevin dynamics into the diffusion models of §2.3 of the survey, and §2.2 shows how the score \\(\\s_\\sigma\\) is learned from samples when it is not known.</p>
      <p>The animation <a href="../../animations.html#a2">A2, Scores and Langevin dynamics</a> runs the same experiment on a spiral mixture with ten levels. It writes the step as \\(h = 2\\eta\\), so its update reads \\(\\x_{k+1} = \\x_k + \\tfrac{h}{2}\\s(\\x_k) + \\sqrt{h}\\,\\bm\\xi_k\\) with \\(h = 0.3\\,\\sigma^2\\), which is the same rule as \\(\\eta = 0.15\\,\\sigma^2\\) here.</p>`,
  },
];
