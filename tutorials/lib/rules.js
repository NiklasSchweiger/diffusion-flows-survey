// The rules and earlier results that the derivation lines use. A derivation line lists them as
//   { tex, why, uses: ['ibp', 'expectation'] }
// and the page shows them as chips. kind 'rule' is general mathematics, explained on the toolbox page
// (../toolbox/#id). kind 'result' is something an earlier step derived; its chip links to that step.
// tex uses the macros that every tutorial page defines (\x, \E, \dd, \diver, \KL, \W, \N, \R, \bm).

export const GROUPS = [
  ['calc', 'Calculus'],
  ['vec', 'Vector calculus'],
  ['prob', 'Probability'],
  ['stoch', 'Stochastic calculus'],
];

export const RULES = {
  // ---------- Calculus ----------
  'log-rules': {
    kind: 'rule', group: 'calc', name: 'Logarithm rules',
    tex: '\\log(ab) = \\log a + \\log b, \\qquad \\log\\frac{a}{b} = \\log a - \\log b, \\qquad \\log e^{c} = c',
    short: 'A log turns products into sums and quotients into differences, and it undoes the exponential.',
    body: [
      'The logarithm is the inverse of the exponential, and \\(e^{u+v} = e^u e^v\\) turns into \\(\\log(ab) = \\log a + \\log b\\). The other two rules follow from this one.',
      'This is why scores are written with a log. A density of the form \\(e^{-E}/Z\\) becomes \\(-E - \\log Z\\), and the normalizing constant separates from everything that depends on \\(\\x\\).',
    ],
    example: '\\log\\frac{e^{-E(\\x)}}{Z} = -E(\\x) - \\log Z',
  },
  'log-deriv': {
    kind: 'rule', group: 'calc', name: 'Derivative of a logarithm',
    tex: '\\nabla\\log f = \\frac{\\nabla f}{f}, \\qquad \\nabla f = f\\,\\nabla\\log f',
    short: 'The gradient of \\(\\log f\\) is the gradient of \\(f\\) divided by \\(f\\), for a positive function \\(f\\).',
    body: [
      'This is the chain rule applied to \\(\\log\\), whose derivative is \\(1/u\\). It holds wherever \\(f > 0\\), which is true for every density on these pages.',
      'Read from right to left, it rewrites a gradient of a density as the density times its score. That form is what lets a derivation write an integral of \\(\\nabla p\\) as an expectation under \\(p\\).',
    ],
    example: '\\int \\bm s\\cdot\\nabla p\\,\\dd\\x = \\int p\\,\\bm s\\cdot\\nabla\\log p\\,\\dd\\x = \\E_p\\big[\\bm s\\cdot\\nabla\\log p\\big]',
  },
  'grad-linear': {
    kind: 'rule', group: 'calc', name: 'Linearity of derivatives',
    tex: '\\nabla(af + bg) = a\\nabla f + b\\nabla g, \\qquad \\nabla c = 0 \\text{ for a constant } c',
    short: 'Derivatives act term by term, constant factors move outside, and a constant has derivative zero.',
    body: [
      'The gradient, the divergence and the time derivative are all linear. A sum can be differentiated term by term, and a factor that does not depend on the variable of differentiation can be moved outside.',
      'Whether something is a constant depends on the variable. \\(Z\\) and \\(g_\\tau^2\\) depend on time but not on \\(\\x\\), so for \\(\\nabla\\) and \\(\\diver\\) they behave like numbers.',
    ],
    example: '\\nabla_{\\x}\\big(-E(\\x) - \\log Z\\big) = -\\nabla E(\\x) - 0',
  },
  'chain-rule': {
    kind: 'rule', group: 'calc', name: 'Chain rule along a path',
    tex: '\\frac{\\dd}{\\dd t}\\,\\varphi(\\x_t) = \\nabla\\varphi(\\x_t)\\cdot\\dot\\x_t',
    short: 'The rate of change of a function along a moving point is its gradient dotted with the velocity.',
    body: [
      'If a point moves along a smooth path \\(\\x_t\\), a function evaluated at that point changes because the point moves. To first order the change over a short time \\(h\\) is \\(\\nabla\\varphi\\cdot(\\x_{t+h} - \\x_t) \\approx h\\,\\nabla\\varphi\\cdot\\dot\\x_t\\).',
      'For a path driven by noise this is no longer enough, because the squared increments do not vanish. Itô’s formula adds the missing second-order term.',
    ],
    example: '\\frac{\\dd}{\\dd t}\\,\\|\\x_t\\|^2 = 2\\,\\x_t\\cdot\\dot\\x_t',
  },
  'diff-under-int': {
    kind: 'rule', group: 'calc', name: 'Differentiating under the integral',
    tex: '\\nabla_{\\x}\\int f(\\x, \\bm z)\\,\\dd\\bm z = \\int \\nabla_{\\x} f(\\x, \\bm z)\\,\\dd\\bm z',
    short: 'A derivative with respect to \\(\\x\\) can move inside an integral over a different variable.',
    body: [
      'The integral is a sum over \\(\\bm z\\), and the derivative of a sum is the sum of the derivatives. The exchange needs the integrand and its derivative to be bounded by an integrable function, which holds for the Gaussian mixtures used here.',
      'The same applies to a time derivative of an integral over space, which is how the time derivative of a KL divergence or of an expectation is computed.',
    ],
    example: '\\nabla p_t(\\x) = \\int \\nabla_{\\x}\\,p_t(\\x\\mid\\x_1)\\,p_{\\mathrm{data}}(\\x_1)\\,\\dd\\x_1',
  },
  'time-change': {
    kind: 'rule', group: 'calc', name: 'Reversing the time variable',
    tex: '\\tau = 1 - t \\;\\Rightarrow\\; \\frac{\\dd}{\\dd\\tau} = -\\frac{\\dd}{\\dd t}, \\qquad \\rho_\\tau = p_{1-\\tau}',
    short: 'With \\(\\tau = 1 - t\\), a derivative in \\(\\tau\\) is minus the derivative in \\(t\\).',
    body: [
      'The survey writes the corruption in forward time \\(\\tau\\), from data to noise, and the generative direction in \\(t\\), from noise to data. The two clocks run in opposite directions, so every time derivative changes sign when you switch between them.',
      'Only the sign of time derivatives changes. Spatial quantities such as \\(\\nabla\\log p\\) and the Laplacian are the same in both clocks.',
    ],
    example: '\\partial_t\\,p_t = -\\partial_\\tau\\,\\rho_\\tau\\big|_{\\tau = 1-t}',
  },
  'linear-ode': {
    kind: 'rule', group: 'calc', name: 'Linear ODEs',
    tex: '\\dot a_\\tau = f_\\tau\\,a_\\tau,\\ a_0 = 1 \\;\\Rightarrow\\; a_\\tau = \\exp\\Big(\\int_0^\\tau f_s\\,\\dd s\\Big), \\qquad f_\\tau = \\frac{\\dot a_\\tau}{a_\\tau}',
    short: 'A quantity whose rate of change is proportional to itself is an exponential of the integrated rate.',
    body: [
      'Dividing by \\(a_\\tau\\) gives \\(\\frac{\\dd}{\\dd\\tau}\\log a_\\tau = f_\\tau\\), and integrating gives the exponential. Read in reverse, the rate is the log derivative of the solution.',
      'A variance obeys a linear ODE with a source term, \\(\\dot v = 2fv + g^2\\). Given \\(v\\) and \\(f\\), the source can be read off as \\(g^2 = \\dot v - 2fv\\).',
    ],
    example: 'a_\\tau = \\alpha_{1-\\tau} \\;\\Rightarrow\\; f_\\tau = \\frac{\\dot a_\\tau}{a_\\tau} = -\\frac{\\dot\\alpha_t}{\\alpha_t}',
  },
  'mixed-partials': {
    kind: 'rule', group: 'calc', name: 'Mixed partial derivatives commute',
    tex: '\\partial_1\\partial_2 f = \\partial_2\\partial_1 f',
    short: 'For a smooth function, the order of two partial derivatives does not matter.',
    body: [
      'This is Schwarz’s theorem. It holds whenever the second partial derivatives are continuous, which is true for every density on these pages.',
      'It is the reason a rotated gradient field has zero divergence: the two terms of the divergence are the same mixed derivative with opposite signs.',
    ],
    example: '\\diver\\big(-\\partial_2 f,\\ \\partial_1 f\\big) = -\\partial_1\\partial_2 f + \\partial_2\\partial_1 f = 0',
  },

  substitution: {
    kind: 'rule', group: 'calc', name: 'Change of variables in an integral',
    tex: '\\int_{a}^{b} f\\big(g(t)\\big)\\,g\'(t)\\,\\dd t = \\int_{g(a)}^{g(b)} f(\\lambda)\\,\\dd\\lambda',
    short: 'An integral over t becomes an integral over \\(\\lambda = g(t)\\) once \\(\\dd t\\) is replaced by \\(\\dd\\lambda / g\'(t)\\).',
    body: [
      'For an increasing \\(g\\), a small interval \\(\\dd t\\) is mapped to an interval of length \\(\\dd\\lambda = g\'(t)\\,\\dd t\\). Summing \\(f\\) over the small intervals in either variable gives the same total.',
      'In the tutorials \\(g\\) is the log signal-to-noise ratio \\(\\lambda_t\\) of a schedule. Rewriting an integral over t as an integral over \\(\\lambda\\) shows which parts of an objective depend on the schedule and which do not.',
    ],
    example: '\\int_0^1 \\dot\\lambda_t\\,\\ell(\\lambda_t)\\,\\dd t = \\int_{\\lambda_{\\min}}^{\\lambda_{\\max}} \\ell(\\lambda)\\,\\dd\\lambda',
  },
  riemann: {
    kind: 'rule', group: 'calc', name: 'Riemann sums',
    tex: '\\sum_{i} f(t_{i-1})\\,\\big(g(t_i) - g(t_{i-1})\\big) \;\\longrightarrow\; \\int_0^1 f(t)\\,g\'(t)\\,\\dd t',
    short: 'A sum of function values times small increments tends to an integral as the grid is refined.',
    body: [
      'Each term is the area of a thin bar of height \\(f(t_{i-1})\\) and width \\(g(t_i) - g(t_{i-1}) \\approx g\'(t_{i-1})(t_i - t_{i-1})\\). When the largest gap of the grid tends to zero, the bars fill the area under \\(f g\'\\).',
      'With a coarse grid the sum can be far from the integral, in particular when \\(g\\) grows fast. That is the case for the signal-to-noise ratio near the data end.',
    ],
    example: '\\sum_i \\tfrac12\\big(\\mathrm{SNR}(t_i) - \\mathrm{SNR}(t_{i-1})\\big)\\,e(t_{i-1}) \\to \\tfrac12\\int_0^1 \\tfrac{\\dd}{\\dd t}\\mathrm{SNR}(t)\\,e(t)\\,\\dd t',
  },
  ftc: {
    kind: 'rule', group: 'calc', name: 'Fundamental theorem of calculus',
    tex: 'f(b) - f(a) = \\int_a^b f\'(s)\\,\\dd s',
    short: 'A function is the integral of its derivative, plus its value at one end.',
    body: [
      'Integrating a rate of change over an interval gives the total change. This turns a statement about a derivative along a path into a statement about the end points of the path.',
    ],
    example: '\\log p_1(\\x_1) - \\log p_0(\\x_0) = \\int_0^1 \\frac{\\dd}{\\dd t}\\log p_t(\\x_t)\\,\\dd t',
  },
  fubini: {
    kind: 'rule', group: 'calc', name: 'Exchanging the order of integration',
    tex: '\\int_{a}^{b}\\!\\int_{\\lambda}^{b} h(\\lambda, \\lambda\')\\,\\dd\\lambda\'\\,\\dd\\lambda = \\int_{a}^{b}\\!\\int_{a}^{\\lambda\'} h(\\lambda, \\lambda\')\\,\\dd\\lambda\\,\\dd\\lambda\'',
    short: 'A double integral over a triangle can be computed in either order, as long as the limits describe the same triangle.',
    body: [
      'Both sides integrate \\(h\\) over the pairs with \\(a \\le \\lambda \\le \\lambda\' \\le b\\). On the left the outer variable is \\(\\lambda\\) and \\(\\lambda\'\\) runs above it; on the right the outer variable is \\(\\lambda\'\\) and \\(\\lambda\\) runs below it. This is Fubini’s theorem, which needs \\(h\\) to be integrable, as it is for every function on these pages.',
    ],
    example: '\\int \\ell(\\lambda)\\int_\\lambda^{\\lambda_{\\max}} p(\\lambda\')\\,\\dd\\lambda\'\\,\\dd\\lambda = \\int p(\\lambda\')\\int_{\\lambda_{\\min}}^{\\lambda\'}\\ell(\\lambda)\\,\\dd\\lambda\\,\\dd\\lambda\'',
  },

  // ---------- Vector calculus ----------
  'div-grad-lap': {
    kind: 'rule', group: 'vec', name: 'Gradient, divergence and Laplacian',
    tex: '\\nabla f = (\\partial_1 f, \\dots, \\partial_d f), \\qquad \\diver\\bm v = \\textstyle\\sum_i \\partial_i v_i, \\qquad \\Delta f = \\diver\\nabla f = \\sum_i \\partial_i^2 f',
    short: 'The Laplacian is the divergence of the gradient.',
    body: [
      'The gradient turns a scalar function into a vector field that points uphill. The divergence turns a vector field into a scalar that measures how much the field spreads out of a point. The Laplacian chains the two.',
      'In one dimension all three are ordinary derivatives: \\(\\nabla f = f^{\\prime}\\), \\(\\diver v = v^{\\prime}\\) and \\(\\Delta f = f^{\\prime\\prime}\\). At a maximum of \\(f\\) the Laplacian is negative.',
    ],
    example: '\\Delta\\rho = \\diver\\nabla\\rho = \\diver\\big(\\rho\\,\\nabla\\log\\rho\\big)',
  },
  ibp: {
    kind: 'rule', group: 'vec', name: 'Integration by parts',
    tex: '\\int_{\\R^d} \\varphi\\;\\diver\\bm v\\;\\dd\\x = -\\int_{\\R^d} \\nabla\\varphi\\cdot\\bm v\\;\\dd\\x \\quad\\text{when } \\varphi\\,\\bm v \\to 0 \\text{ at infinity}',
    short: 'A derivative can move from one factor of an integrand to the other, at the cost of a minus sign.',
    body: [
      'The product rule gives \\(\\diver(\\varphi\\bm v) = \\nabla\\varphi\\cdot\\bm v + \\varphi\\,\\diver\\bm v\\). The integral of a divergence over all of \\(\\R^d\\) is the flux through a sphere at infinity, which is zero when \\(\\varphi\\bm v\\) decays. Integrating the product rule gives the formula.',
      'Derivations use it to move derivatives off something unknown, such as the data score, or off a test function onto a density. Applied twice, it moves a Laplacian: \\(\\int\\varphi\\,\\Delta\\rho = \\int\\rho\\,\\Delta\\varphi\\).',
    ],
    example: '\\int \\bm s_\\theta\\cdot\\nabla p\\,\\dd\\x = -\\int p\\,\\diver\\bm s_\\theta\\,\\dd\\x',
  },
  'test-functions': {
    kind: 'rule', group: 'vec', name: 'Equality through test functions',
    tex: '\\int \\varphi\\,f\\,\\dd\\x = \\int \\varphi\\,g\\,\\dd\\x \\text{ for every smooth } \\varphi \\;\\Rightarrow\\; f = g',
    short: 'Two functions that have the same integral against every smooth test function are equal.',
    body: [
      'If \\(f\\) and \\(g\\) differed near some point, a test function concentrated there would give different integrals. So agreement for all test functions forces \\(f = g\\).',
      'This is how equations for densities are derived. One computes how \\(\\E[\\varphi(\\x_t)]\\) changes in time, which only needs the path of one sample, and then removes \\(\\varphi\\) to get an equation for the density itself.',
    ],
    example: '\\int \\varphi\\,\\partial_t p_t = -\\int \\varphi\\,\\diver(p_t\\bm u_t) \\text{ for all } \\varphi \\;\\Rightarrow\\; \\partial_t p_t = -\\diver(p_t\\bm u_t)',
  },

  // ---------- Probability ----------
  expectation: {
    kind: 'rule', group: 'prob', name: 'Expectation as an integral',
    tex: '\\E\\big[f(\\x)\\big] = \\int f(\\x)\\,p(\\x)\\,\\dd\\x, \\qquad \\E[aX + bY] = a\\,\\E[X] + b\\,\\E[Y]',
    short: 'An expectation is an integral against the density, and it is linear.',
    body: [
      'For a random vector with density \\(p\\), the expectation of a function is its average weighted by \\(p\\). Because it is an integral, it is linear, and the rules of calculus such as integration by parts apply to it.',
      'Linearity holds even when \\(X\\) and \\(Y\\) are dependent. It is what lets a derivation expand a square inside an expectation and take the expectation term by term.',
    ],
    example: '\\E_p\\big[\\bm s\\cdot\\nabla\\log p\\big] = \\int p\\;\\bm s\\cdot\\frac{\\nabla p}{p}\\,\\dd\\x = \\int \\bm s\\cdot\\nabla p\\,\\dd\\x',
  },
  'expand-square': {
    kind: 'rule', group: 'prob', name: 'Expanding a squared norm',
    tex: '\\|\\bm a - \\bm b\\|^2 = \\|\\bm a\\|^2 - 2\\,\\bm a\\cdot\\bm b + \\|\\bm b\\|^2',
    short: 'The squared distance between two vectors splits into two squared lengths and a cross term.',
    body: [
      'Write \\(\\|\\bm a - \\bm b\\|^2 = (\\bm a - \\bm b)\\cdot(\\bm a - \\bm b)\\) and multiply out. The same holds for a sum: \\(\\|\\bm a + \\bm b\\|^2 = \\|\\bm a\\|^2 + 2\\,\\bm a\\cdot\\bm b + \\|\\bm b\\|^2\\).',
      'A common move is to add and subtract a well-chosen vector \\(\\bm m\\) first, \\(\\bm a - \\bm b = (\\bm a - \\bm m) + (\\bm m - \\bm b)\\), and then expand. The derivation is set up so that the cross term vanishes.',
    ],
    example: '\\|f - Y\\|^2 = \\|f - m\\|^2 + \\|m - Y\\|^2 + 2\\,(f - m)\\cdot(m - Y)',
  },
  'cond-exp': {
    kind: 'rule', group: 'prob', name: 'Conditional expectation',
    tex: '\\E\\big[\\,\\E[Y\\mid X]\\,\\big] = \\E[Y], \\qquad \\E\\big[g(X)\\,Y\\mid X\\big] = g(X)\\,\\E[Y\\mid X]',
    short: 'Averaging in two stages gives the full average, and a function of \\(X\\) is a constant given \\(X\\).',
    body: [
      '\\(\\E[Y\\mid X]\\) is the average of \\(Y\\) over all outcomes that share the observed value of \\(X\\). It is itself a function of \\(X\\). Averaging it over \\(X\\) gives back \\(\\E[Y]\\), the tower property.',
      'Given \\(X\\), anything computed from \\(X\\) alone is fixed and can be pulled out of the conditional expectation. Also \\(\\E[X\\mid X] = X\\). Together these show that a residual \\(Y - \\E[Y\\mid X]\\) is uncorrelated with every function of \\(X\\).',
    ],
    example: '\\E\\big[\\nabla\\varphi(\\x_t)\\cdot\\dot\\x_t\\big] = \\E\\big[\\nabla\\varphi(\\x_t)\\cdot\\E[\\dot\\x_t\\mid\\x_t]\\big]',
  },
  bayes: {
    kind: 'rule', group: 'prob', name: 'Marginals and Bayes’ rule',
    tex: 'p(\\x) = \\int p(\\x\\mid\\bm z)\\,p(\\bm z)\\,\\dd\\bm z, \\qquad p(\\bm z\\mid\\x) = \\frac{p(\\x\\mid\\bm z)\\,p(\\bm z)}{p(\\x)}',
    short: 'A marginal averages the conditionals, and the posterior is the joint density divided by the marginal.',
    body: [
      'The joint density is \\(p(\\x\\mid\\bm z)\\,p(\\bm z)\\). Integrating out \\(\\bm z\\) gives the marginal of \\(\\x\\). Dividing the joint by that marginal gives the conditional density of \\(\\bm z\\) given \\(\\x\\), which is Bayes’ rule.',
      'In the tutorials \\(\\bm z\\) is the clean data point \\(\\x_1\\) and \\(\\x\\) is the noisy point \\(\\x_t\\). The posterior says which data points could have produced what we see.',
    ],
    example: 'p_t(\\x_1\\mid\\x) = \\frac{p_t(\\x\\mid\\x_1)\\,p_{\\mathrm{data}}(\\x_1)}{p_t(\\x)}',
  },
  gauss: {
    kind: 'rule', group: 'prob', name: 'Gaussian log-density and its gradient',
    tex: '\\log\\N(\\x;\\bm m,\\sigma^2 I) = -\\frac{\\|\\x - \\bm m\\|^2}{2\\sigma^2} + \\text{const}, \\qquad \\nabla_{\\x}\\log\\N = -\\frac{\\x - \\bm m}{\\sigma^2}',
    short: 'The score of an isotropic Gaussian points from \\(\\x\\) back to the mean, scaled by one over the variance.',
    body: [
      'The density is \\((2\\pi\\sigma^2)^{-d/2}\\exp\\big(-\\|\\x - \\bm m\\|^2/2\\sigma^2\\big)\\). The log removes the exponential, and the prefactor becomes a constant that does not depend on \\(\\x\\).',
      'The gradient of \\(\\tfrac12\\|\\x - \\bm m\\|^2\\) is \\(\\x - \\bm m\\), which gives the score. It is linear in \\(\\x\\) and zero at the mean.',
    ],
    example: '\\nabla\\log p_t(\\x\\mid\\x_1) = -\\frac{\\x - \\alpha_t\\x_1}{\\sigma_t^2}',
  },
  kl: {
    kind: 'rule', group: 'prob', name: 'KL divergence',
    tex: '\\KL(\\rho\\,\\|\\,p) = \\int \\rho\\,\\log\\frac{\\rho}{p}\\,\\dd\\x \\;\\ge\\; 0, \\quad\\text{with equality only if } \\rho = p',
    short: 'The KL divergence measures how far one density is from another; it is zero only when they agree.',
    body: [
      'It is the expected log ratio of the two densities under \\(\\rho\\). It is not symmetric and not a distance, but it is non-negative, which follows from Jensen’s inequality applied to the concave logarithm.',
      'Showing that a dynamics makes \\(\\KL(\\rho_\\tau\\|p)\\) decrease is a standard way to show that \\(\\rho_\\tau\\) converges to \\(p\\).',
    ],
    example: '\\frac{\\dd}{\\dd\\tau}\\KL(\\rho_\\tau\\|p) = \\int \\partial_\\tau\\rho_\\tau\\,\\log\\frac{\\rho_\\tau}{p}\\,\\dd\\x',
  },

  jensen: {
    kind: 'rule', group: 'prob', name: 'Jensen’s inequality',
    tex: '\\log\\E[Z] \;\\ge\; \\E[\\log Z] \\quad\\text{for a positive random variable } Z',
    short: 'The log of an average is at least the average of the logs, because the logarithm is concave.',
    body: [
      'A concave function lies above each of its chords, so it maps an average to a value at least as large as the average of the mapped values. Equality holds only if \\(Z\\) is constant.',
      'Applied to an importance weight \\(Z = p(\\x, \\bm z)/q(\\bm z)\\) with \\(\\bm z \\sim q\\), it gives the evidence lower bound: \\(\\log p(\\x) = \\log\\E_q[Z] \\ge \\E_q[\\log Z]\\). The gap is \\(\\KL\\big(q(\\bm z)\\,\\|\\,p(\\bm z\\mid\\x)\\big)\\), so the bound is tight when \\(q\\) is the true posterior.',
    ],
    example: '\\log p_\\theta(\\x_1) \\ge \\E_q\\Big[\\log\\frac{p_\\theta(\\x_1, \\x_{0:N-1})}{q(\\x_{0:N-1}\\mid\\x_1)}\\Big]',
  },
  'gauss-kl': {
    kind: 'rule', group: 'prob', name: 'KL between Gaussians of equal variance',
    tex: '\\KL\\big(\\N(\\bm m_1, \\varsigma^2 I)\\,\\|\\,\\N(\\bm m_2, \\varsigma^2 I)\\big) = \\frac{\\|\\bm m_1 - \\bm m_2\\|^2}{2\\varsigma^2}',
    short: 'Two Gaussians with the same covariance differ only through their means, and their KL is the squared distance of the means over twice the variance.',
    body: [
      'The log ratio of the two densities is \\(\\big(\\|\\x - \\bm m_2\\|^2 - \\|\\x - \\bm m_1\\|^2\\big)/2\\varsigma^2\\); the normalizing constants cancel. Averaging over \\(\\x \\sim \\N(\\bm m_1, \\varsigma^2 I)\\) and expanding the squares leaves \\(\\|\\bm m_1 - \\bm m_2\\|^2/2\\varsigma^2\\).',
    ],
    example: '\\KL\\big(q(\\x_t\\mid\\x_s,\\x_1)\\,\\|\\,p_\\theta(\\x_t\\mid\\x_s)\\big) = \\frac{\\|\\bm\\mu(\\x_s,\\x_1) - \\bm\\mu(\\x_s,\\hat\\x_1)\\|^2}{2\\varsigma^2}',
  },

  // ---------- Stochastic calculus ----------
  brownian: {
    kind: 'rule', group: 'stoch', name: 'Brownian increments',
    tex: '\\dd\\W \\sim \\N(0,\\,\\dd\\tau\\,I), \\qquad \\E[\\dd\\W] = 0, \\qquad (\\dd\\W)^2 = \\dd\\tau',
    short: 'A Brownian increment has mean zero and variance equal to the time step, independent of the past.',
    body: [
      'Over a step of length \\(h\\) the increment is Gaussian with variance \\(h\\), so its size is of order \\(\\sqrt h\\) rather than \\(h\\). That is why its square, of order \\(h\\), survives in a Taylor expansion and its higher powers do not.',
      'Because each increment is independent of the past and has mean zero, a sum of terms of the form \\(\\text{(something known now)}\\cdot\\dd\\W\\) has expectation zero. Such a sum is called a martingale.',
    ],
    example: '\\E\\Big[\\int_0^\\tau g_s\\,\\nabla\\varphi(\\y_s)\\cdot\\dd\\W_s\\Big] = 0',
  },
  ito: {
    kind: 'rule', group: 'stoch', name: 'Itô’s formula',
    tex: '\\dd\\y = \\bm b\\,\\dd\\tau + g\\,\\dd\\W \\;\\Rightarrow\\; \\dd\\varphi(\\y) = \\Big(\\nabla\\varphi\\cdot\\bm b + \\tfrac12 g^2\\,\\Delta\\varphi\\Big)\\dd\\tau + g\\,\\nabla\\varphi\\cdot\\dd\\W',
    short: 'The chain rule for a noisy path has an extra second-order term, because \\((\\dd\\W)^2 = \\dd\\tau\\).',
    body: [
      'Expand \\(\\varphi(\\y + \\dd\\y)\\) to second order. The first-order term gives \\(\\nabla\\varphi\\cdot\\dd\\y\\). In the second-order term, \\((\\dd\\W)^2 = \\dd\\tau\\) is of the same order as \\(\\dd\\tau\\), so \\(\\tfrac12 g^2\\Delta\\varphi\\,\\dd\\tau\\) remains. All other products are of smaller order.',
      'Without noise, \\(g = 0\\), the formula reduces to the ordinary chain rule along a path.',
    ],
    example: '\\dd\\,\\|\\y\\|^2 = \\big(2\\,\\y\\cdot\\bm b + g^2 d\\big)\\,\\dd\\tau + 2g\\,\\y\\cdot\\dd\\W',
  },

  // ---------- Results derived in the tutorials ----------
  'fokker-planck': {
    kind: 'result', name: 'Fokker–Planck equation', at: '§2.3, step 2', href: '../sde/#fp',
    tex: '\\partial_\\tau\\rho_\\tau = -\\diver(\\bm b\\,\\rho_\\tau) + \\tfrac12 g_\\tau^2\\,\\Delta\\rho_\\tau',
    short: 'How the density of an SDE with drift \\(\\bm b\\) and diffusion \\(g\\) changes in time.',
  },
  transport: {
    kind: 'result', name: 'Diffusion as transport along the score', at: '§2.3, step 3', href: '../sde/#transport',
    tex: '\\tfrac12 g^2\\,\\Delta\\rho = \\diver\\big(\\tfrac12 g^2\\,\\rho\\,\\nabla\\log\\rho\\big)',
    short: 'The diffusion term of Fokker–Planck is a transport term with velocity \\(-\\tfrac12 g^2\\nabla\\log\\rho\\).',
  },
  'forward-sde': {
    kind: 'result', name: 'Coefficients of the forward SDE', at: '§2.3, step 1', href: '../sde/#forward',
    tex: 'f_\\tau = -\\frac{\\dot\\alpha_t}{\\alpha_t}, \\qquad g_\\tau^2 = \\sigma_t^2\\,\\dot\\lambda_t, \\qquad t = 1 - \\tau',
    short: 'The drift and diffusion coefficient that give the interpolant \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\).',
  },
  'pf-ode': {
    kind: 'result', name: 'Probability-flow ODE', at: '§2.3, step 4', href: '../sde/#pfode',
    tex: '\\frac{\\dd\\y}{\\dd\\tau} = \\bm b(\\y) - \\tfrac12 g_\\tau^2\\,\\nabla\\log\\rho_\\tau(\\y)',
    short: 'A deterministic ODE with the same marginals as the forward SDE.',
  },
  'reverse-sde': {
    kind: 'result', name: 'Reverse-time SDE', at: '§2.3, step 5', href: '../sde/#reverse',
    tex: '\\dd\\x = \\big(-\\bm b + g^2\\,\\nabla\\log p_t(\\x)\\big)\\,\\dd t + g\\,\\dd\\W',
    short: 'An SDE that runs from noise to data with the same marginals as the forward process.',
  },
  'cond-score': {
    kind: 'result', name: 'Conditional score', at: '§2.2, step 3', href: '../score-matching/#cond',
    tex: '\\nabla\\log p_t(\\x_t\\mid\\x_1) = -\\frac{\\x_t - \\alpha_t\\x_1}{\\sigma_t^2} = -\\frac{\\x_0}{\\sigma_t}',
    short: 'The score of one noisy data point is the noise divided by \\(-\\sigma_t\\).',
  },
  'regression-lemma': {
    kind: 'result', name: 'Regression lemma', at: '§2.2, step 4', href: '../score-matching/#dsm',
    tex: '\\E\\|f(X) - Y\\|^2 = \\E\\|f(X) - \\E[Y\\mid X]\\|^2 + \\E\\|Y - \\E[Y\\mid X]\\|^2',
    short: 'Regressing on a noisy target is minimized by the conditional mean of the target.',
  },
  'marginal-score': {
    kind: 'result', name: 'Marginal score', at: '§2.2, step 5', href: '../score-matching/#marginal',
    tex: '\\nabla\\log p_t(\\x) = -\\frac{1}{\\sigma_t}\\,\\E[\\x_0\\mid\\x_t = \\x]',
    short: 'The score of the noisy marginal is the average conditional score given \\(\\x_t = \\x\\).',
  },
  tweedie: {
    kind: 'result', name: 'Tweedie’s formula', at: '§2.2, step 6', href: '../score-matching/#tweedie',
    tex: '\\E[\\x_1\\mid\\x_t = \\x] = \\frac{\\x + \\sigma_t^2\\,\\nabla\\log p_t(\\x)}{\\alpha_t}, \\qquad \\E[\\x_0\\mid\\x_t = \\x] = -\\sigma_t\\nabla\\log p_t(\\x)',
    short: 'The best denoiser is a score: the posterior means of data and noise are linear in the score.',
  },
  continuity: {
    kind: 'result', name: 'Continuity equation', at: '§2.4, step 1', href: '../flow-matching/#continuity',
    tex: '\\partial_t p_t + \\diver(p_t\\,\\bm u_t) = 0',
    short: 'A velocity field \\(\\bm u_t\\) generates the path \\(p_t\\) when probability mass is conserved along the flow.',
  },
  'elbo-step': {
    kind: 'result', name: 'One step of the ELBO', at: '§2.5, step 2', href: '../likelihood/#kl',
    tex: '\\KL\\big(q(\\x_t\\mid\\x_s,\\x_1)\\,\\|\\,p_\\theta(\\x_t\\mid\\x_s)\\big) = \\tfrac12\\big(\\mathrm{SNR}(t) - \\mathrm{SNR}(s)\\big)\\,\\|\\x_1 - \\hat\\x_1(\\x_s, s)\\|^2',
    short: 'Each denoising term of the bound is a data-prediction error, weighted by the signal-to-noise ratio the step adds.',
  },
};
