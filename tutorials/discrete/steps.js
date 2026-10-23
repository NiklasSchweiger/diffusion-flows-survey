// Content of the §2.9 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex,
// section "Discrete and language diffusion"; S(id) tags a symbol from symbols.js, CHG marks the part
// of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const t = S('t'), pt = S('pt'), R = S('R'), Q = S('Q'), r = S('r'), s = S('s'), al = S('alpha'), pth = S('ptheta'), x1 = S('x1'), xt = S('xt');
const ka = S('kappa'), I = S('I');

export const INTRO = `
  <div class="kicker">Survey §2.9 · Interactive tutorial</div>
  <h1>Discrete and language diffusion</h1>
  <p>Text, code and the tokens of quantized images take values in a finite set. There is no Gaussian noise and no gradient there, but the structure of the earlier tutorials carries over almost line by line once the stochastic differential equation is replaced by a continuous-time Markov chain. This tutorial follows the survey through that correspondence: the forward equation of a chain, its time reversal and the concrete score, the score-entropy loss, the ELBO of masked diffusion, discrete flow matching, and the price of revealing many tokens at once.</p>
  <p>No step trains a network. The chains are small enough that every marginal has a closed form, and the last three steps use the toy language of Animation A4: 64 sentences of eight tokens with three coupled pairs of positions and known probabilities, so the posterior given any partly masked sentence is computed exactly by enumeration. The convention is that of the survey: the corruption runs from data at \\(t = 1\\) to noise at \\(t = 0\\), and the generative chain runs forward in \\(t\\).</p>
  <ol class="toc">
    <li><a href="#chain">A Markov chain on a finite set</a></li>
    <li><a href="#reverse">Time reversal and the concrete score</a></li>
    <li><a href="#entropy">Score entropy</a></li>
    <li><a href="#elbo">Masked diffusion and its ELBO</a></li>
    <li><a href="#dfm">Discrete flow matching: masked and uniform sources</a></li>
    <li><a href="#parallel">Parallel decoding and its price</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'chain', kicker: 'The discrete Fokker–Planck equation', title: 'A Markov chain on a finite set', scene: 'chain',
    prose: [
      'A Markov chain on a finite set \\(\\mathcal X\\) is described by rates: the probability of a jump from x to y in a short interval is \\(R_t(x, y)\\,dt\\). The marginal \\(p_t\\) then changes by what flows into each state minus what flows out of it. This is the Kolmogorov forward equation, which plays the role of the Fokker–Planck equation of §2.5.',
      'The state space has five words and the mask symbol [M]. Going down from \\(t = 1\\), each token is kept until a uniform random time and then replaced: by a uniform draw over the five words (the uniform kernel of D3PM) or by [M] (the absorbing kernel). The marginals are then \\(p_t = t\\,p_1 + (1 - t)/K\\) and \\(p_t = t\\,p_1\\) with mass \\(1 - t\\) on [M]. The filled bars are these closed forms, the dots are the histogram of 400 seeded chains, and the outlined bars are \\(p_1\\). Click a word to see its inflow and outflow under the generative rates of step 2.',
    ],
    tex: `\\partial_t ${pt}(y) = \\sum_{x \\ne y}\\big(${pt}(x)\\,${R}(x, y) - ${pt}(y)\\,${R}(y, x)\\big)`,
    after: ['Under the absorbing kernel nothing flows out of a word as t grows: the generative chain only turns [M] into words, at the total rate \\(p_1(y)/(1 - t)\\) towards word y. Under the uniform kernel words flow into each other in both directions, and the net flow is the small difference of two rates that grow like \\(1/t\\) near the noise end.'],
    check: {
      q: 'Under the uniform kernel with K = 5 words, what probability does each word have at \\(t = 0\\)?',
      answer: 0.2, tol: 0.005,
      fb: { true: 'Right. At t = 0 every token has been replaced by a uniform draw, so \\(p_0 = 1/K = 0.2\\) for every word, whatever \\(p_1\\) was. The absorbing kernel instead puts all mass on [M], which has no outflow under the corruption.', false: 'Set t = 0 in \\(p_t = t\\,p_1 + (1 - t)/K\\). The data term vanishes and every word has probability \\(1/K\\).' },
    },
  },
  {
    id: 'reverse', kicker: 'Running the corruption backwards', title: 'Time reversal and the concrete score', scene: 'reverse',
    prose: [
      'Write \\(Q_t(y, x)\\) for the rate at which the corruption jumps from y to x as t decreases. The generative chain has to make the jump from x to y exactly as often as the corruption, followed backwards in time, makes the jump from y to x. This fixes its rates, and the only unknown in them is the ratio \\(p_t(y)/p_t(x)\\), which is called the concrete score. It is the discrete counterpart of the score in the reverse-time SDE.',
      'The left panel has 32 states on a lattice with a two-bump \\(p_1\\) and the uniform kernel. The bars are \\(p_t\\), and the dots are the histogram of 400 chains that start from the uniform distribution near \\(t = 0\\) and jump with the rates of the formula. Press Play or move t: the histogram follows \\(p_t\\) to the data. The right panel shows the concrete score towards the right neighbour, placed halfway between x and x + 1, and the curve \\(1 + \\partial_x\\log p_t\\) of the smooth density behind the lattice.',
    ],
    tex: `${R}(x, y) = ${Q}(y, x)\\,${S('cs')}, \\qquad \\frac{p_t(x + ${S('delta')})}{p_t(x)} \\approx 1 + ${S('delta')}\\,\\partial_x \\log p_t(x)`,
    deriv: {
      have: ['The generative chain runs forward in t with rates \\(R_t\\); the corruption runs backward in t with rates \\(Q_t\\).', 'Both describe the same joint law of the states at two nearby times.'],
      goal: 'R_t(x, y) = Q_t(y, x)\\,p_t(y)/p_t(x)',
      plan: 'Write the probability of seeing x at t and y at t + dt once with each chain and set the two expressions equal.',
      title: 'Why the reverse rates involve the concrete score', lines: [
        { tex: `P(X_t = x,\\ X_{t + dt} = y) = ${CHG('p_t(x)\\,R_t(x, y)\\,dt')}`, why: 'The generative chain is at x with probability \\(p_t(x)\\) and jumps to y in the next interval with probability \\(R_t(x, y)\\,dt\\).', uses: ['chain-rule'] },
        { tex: `P(X_{t + dt} = y,\\ X_t = x) = ${CHG('p_t(y)\\,Q_t(y, x)\\,dt')} + o(dt)`, why: 'The corruption is at y at time t + dt and jumps to x as time decreases to t. Replacing \\(p_{t + dt}\\) by \\(p_t\\) changes the product only at order \\(dt^2\\).', uses: ['chain-rule'] },
        { tex: `p_t(x)\\,R_t(x, y) = p_t(y)\\,Q_t(y, x) \\quad\\Rightarrow\\quad R_t(x, y) = ${CHG('Q_t(y, x)\\,\\frac{p_t(y)}{p_t(x)}')}`, why: 'The two expressions describe the same event, so they are equal. Divide by \\(p_t(x)\\).', uses: ['bayes'] },
      ],
    },
    after: ['The ratio for a step of one lattice site is the exponential of the average of \\(\\partial_x\\log p_t\\) over the step, so it agrees with \\(1 + \\partial_x\\log p_t\\) to first order and departs from it where the log-density is steep. A model of the concrete score is therefore a model of the score, sampled on the lattice.'],
    check: {
      q: 'At some state the log-density falls with slope \\(\\partial_x\\log p_t = -0.3\\) per lattice site. About how large is \\(p_t(x + 1)/p_t(x)\\)?',
      answer: 0.7, tol: 0.05,
      fb: { true: 'Right. To first order the ratio is \\(1 + \\delta\\,\\partial_x\\log p_t = 1 - 0.3 = 0.7\\). The exact value for a constant slope is \\(e^{-0.3} \\approx 0.74\\).', false: 'Use \\(p_t(x + \\delta)/p_t(x) \\approx 1 + \\delta\\,\\partial_x\\log p_t(x)\\) with \\(\\delta = 1\\) and slope −0.3.' },
    },
  },
  {
    id: 'entropy', kicker: 'Learning the concrete score', title: 'Score entropy', scene: 'entropy',
    prose: [
      'The concrete score \\(p_t(y)/p_t(x)\\) is unknown, but the same ratio for the corruption started from one clean value, \\(r = p_t(y\\mid x_1)/p_t(x\\mid x_1)\\), is known in closed form. SEDD regresses on it with the loss \\(s - r\\log s\\) in place of the squared error. This is the move of denoising score matching in §2.3: regress a tractable conditional quantity and obtain the intractable marginal one.',
      'Take the lattice of step 2, a state x and its right neighbour \\(y = x + 1\\). Under the uniform kernel r takes only three values: large when \\(x_1 = y\\), small when \\(x_1 = x\\), and 1 for every other clean value. The bars show the posterior of \\(x_1\\) given x, grouped by these three cases. The curve is the expected loss as a function of s, shifted so that its minimum is zero. Drag the marker along the curve and move t and x.',
    ],
    tex: `\\min_{${s}}\\ \\E_{x_1\\mid x}\\big[${s} - ${r}\\log ${s}\\big], \\qquad ${r} = \\frac{p_t(y\\mid x_1)}{p_t(x\\mid x_1)}`,
    deriv: {
      have: ['The posterior of the clean value given the corrupted state, \\(p(x_1\\mid x) = p_1(x_1)\\,p_t(x\\mid x_1)/p_t(x)\\).', 'The marginal \\(p_t(y) = \\sum_{x_1}p_1(x_1)\\,p_t(y\\mid x_1)\\).'],
      goal: '\\arg\\min_s \\E[s - r\\log s] = p_t(y)/p_t(x)',
      plan: 'Minimize \\(s - \\bar r\\log s\\) for the posterior mean \\(\\bar r\\) of r, then compute that mean with Bayes\' rule.',
      title: 'Why score entropy recovers the marginal ratio', lines: [
        { tex: `\\E[s - r\\log s] = s - \\bar r\\log s, \\quad \\frac{d}{ds}: 1 - \\frac{\\bar r}{s} = 0 \\;\\Rightarrow\\; s = ${CHG('\\bar r = \\E[r\\mid x]')}`, why: 'The loss is linear in r, so only the posterior mean of r matters. The function is convex in s, and its derivative vanishes at \\(s = \\bar r\\).', uses: ['expectation'] },
        { tex: `\\bar r = \\sum_{x_1}\\frac{p_1(x_1)\\,p_t(x\\mid x_1)}{p_t(x)}\\cdot\\frac{p_t(y\\mid x_1)}{p_t(x\\mid x_1)} = ${CHG('\\sum_{x_1}\\frac{p_1(x_1)\\,p_t(y\\mid x_1)}{p_t(x)}')}`, why: 'Write the posterior with Bayes\' rule. The factor \\(p_t(x\\mid x_1)\\) cancels.', uses: ['bayes'] },
        { tex: `\\bar r = ${CHG('\\frac{p_t(y)}{p_t(x)}')}`, why: 'The remaining sum over the clean value is the marginal at y.', uses: ['cond-exp'] },
      ],
    },
    after: ['The excess loss \\(s - \\bar r\\log s - (\\bar r - \\bar r\\log\\bar r)\\) is a Bregman divergence between s and \\(\\bar r\\), so the argument of the regression lemma goes through with the squared error replaced by this divergence. The loss also keeps s positive, which a squared error on a ratio would not.'],
    check: {
      q: 'What does a network trained with score entropy learn at the state x?',
      opts: [['The posterior mean of r, which is the marginal ratio \\(p_t(y)/p_t(x)\\)', true], ['The value of r for the most probable clean value', false], ['The ratio of the data probabilities \\(p_1(y)/p_1(x)\\)', false]],
      fb: { true: 'Right. The loss depends on r only through its posterior mean, and Bayes\' rule turns that mean into the marginal ratio at time t. The vertical line marks it, and the curve has its minimum there.', false: 'The minimizer of \\(s - \\bar r\\log s\\) is \\(\\bar r\\), the average of r over the posterior of \\(x_1\\), and that average equals \\(p_t(y)/p_t(x)\\) at time t, not at t = 1.' },
    },
  },
  {
    id: 'elbo', kicker: 'The absorbing kernel for language', title: 'Masked diffusion and its ELBO', scene: 'elbo',
    prose: [
      'With the absorbing kernel each of the L tokens is independently kept with probability \\(\\alpha_t\\) and replaced by [M] otherwise. The generative chain unmasks a masked token in \\([t, t + dt]\\) with probability \\(\\dot\\alpha_t\\,dt/(1 - \\alpha_t)\\), draws its value from a predictor \\(p_\\theta(x^i\\mid\\x_t)\\), and never changes it afterwards. MDLM and MD4 showed that the ELBO is then a weighted cross-entropy on the masked tokens.',
      'The chips are a sentence of the toy language. The plot shows the integrand of the bound against t for three schedules \\(\\alpha_t\\), and the shaded areas are the bounds. The expectation over masks is computed exactly from all 2⁸ mask patterns. The predictor is the exact conditional of the language at temperature τ. Move τ and switch the sentence: the three areas stay equal, and at τ = 1 they equal \\(-\\log p(\\x_1)\\).',
    ],
    tex: `-\\log ${pth}(${x1}) \\le \\int_0^1 \\frac{\\dot{${al}}}{1 - ${al}}\\;\\E_{${xt}}\\Big[\\sum_{i:\\,x_t^i = ${S('M')}} -\\log ${pth}(x_1^i\\mid ${xt})\\Big]\\,dt`,
    deriv: {
      have: ['Each token is masked independently with probability \\(m = 1 - \\alpha_t\\).', 'Write \\(C(S)\\) for the sum of \\(-\\log p_\\theta(x_1^i\\mid\\x_t)\\) over the masked set S. It depends on S but not on t.'],
      goal: '\\text{bound} = \\sum_S C(S)\\,B(|S|,\\ L + 1 - |S|)',
      plan: 'Write the expectation over masks as a sum over patterns, substitute m for t, and integrate each pattern\'s weight.',
      title: 'Why the bound does not depend on the schedule', lines: [
        { tex: `\\E_{\\x_t}[\\cdots] = \\sum_S ${CHG('m^{|S|}(1 - m)^{L - |S|}')}\\,C(S)`, why: 'Each pattern S has probability \\(m^{|S|}(1 - m)^{L - |S|}\\) because the tokens are masked independently.', uses: ['expectation'] },
        { tex: `\\int_0^1\\frac{\\dot\\alpha_t}{1 - \\alpha_t}(\\cdots)\\,dt = \\sum_S C(S)\\int_0^1 ${CHG('m^{|S| - 1}(1 - m)^{L - |S|}\\,dm')}`, why: 'Substitute \\(m = 1 - \\alpha_t\\), so \\(dm = -\\dot\\alpha_t\\,dt\\) and the weight becomes \\(dm/m\\). The limits \\(\\alpha_0 = 0\\) and \\(\\alpha_1 = 1\\) are the same for every schedule, so the schedule drops out.', uses: ['substitution'] },
        { tex: `\\int_0^1 m^{k - 1}(1 - m)^{L - k}\\,dm = ${CHG('\\frac{(k - 1)!\\,(L - k)!}{L!}')} = \\frac{1}{k\\binom{L}{k}}`, why: 'A Beta integral. It is the probability that, in a uniformly random order, the masked set just before revealing one particular token of S is exactly S.', uses: ['substitution'] },
      ],
    },
    after: ['The last line says that the bound is the chain rule \\(-\\log p_\\theta(\\x_1) = \\sum_j -\\log p_\\theta(x^{\\sigma_j}\\mid x^{\\sigma_1}, \\dots, x^{\\sigma_{j-1}})\\) averaged over uniformly random orders σ. A masked diffusion model is an order-agnostic autoregressive model. For exact conditionals every order gives the same product, so the bound is tight. Averaged over the language the bound is the entropy of the language plus a KL divergence, which is zero only at τ = 1, and the readout shows both. The optimal predictor does not depend on t, because the posterior of a masked token given the visible ones does not.'],
    check: {
      q: 'At τ = 1 the predictor is the exact conditional. What is the gap between the bound and \\(-\\log p(\\x_1)\\), in nats?',
      answer: 0, tol: 0.002,
      fb: { true: 'Right. With exact conditionals the chain rule holds in every order, so the average over orders is exactly \\(-\\log p(\\x_1)\\). The readout shows a gap of zero to rounding error for each sentence.', false: 'Each order gives the chain rule of the exact model, whose product is \\(p(\\x_1)\\). The average of identical values has no gap, so the answer is 0.' },
    },
  },
  {
    id: 'dfm', kicker: 'Probability paths for tokens', title: 'Discrete flow matching: masked and uniform sources', scene: 'dfm',
    prose: [
      'Discrete flow matching starts from the flow-matching side. It fixes a path for each token, \\(p_t(x^i\\mid\\x_1) = \\kappa_t\\,\\delta_{x_1^i} + (1 - \\kappa_t)\\,p_0\\), writes down rates that generate it for one clean sentence, and marginalizes them over the posterior of \\(\\x_1\\) given \\(\\x_t\\), as §2.4 marginalizes conditional velocities. With \\(p_0\\) a point mass on [M] this is masked diffusion. With \\(p_0\\) uniform over the vocabulary it gives a uniform-state model.',
      'Here \\(\\kappa_t = t\\), the vocabulary has the 26 words of the toy language, and the posterior over the 64 sentences is computed exactly for every state. An Euler step of size \\(h = 1/N\\) redraws each token from its posterior marginal with probability \\(h/(1 - t)\\). The table shows the selected run step by step, with changed tokens marked, and the list shows the eight runs. Switch the source and move N.',
    ],
    tex: `\\text{rate of } x_t^i \\to y \\;=\\; \\frac{\\dot{${ka}}}{1 - ${ka}}\\;P\\big(x_1^i = y\\mid ${xt}\\big), \\qquad p_t(x^i\\mid\\x_1) = ${ka}\\,\\delta_{x_1^i} + (1 - ${ka})\\,${S('p0')}`,
    deriv: {
      have: ['The conditional path of one token, \\(p_t(\\cdot\\mid\\x_1) = \\kappa_t\\,\\delta_{x_1^i} + (1 - \\kappa_t)\\,p_0\\).', 'Marginalizing conditional rates with the posterior gives rates for the marginal path, as for velocities in §2.4.'],
      goal: '\\text{rate to } y = \\frac{\\dot\\kappa_t}{1 - \\kappa_t}P(x_1^i = y\\mid\\x_t)',
      plan: 'Find a jump rate that moves the conditional path, then average it over the posterior of the clean sentence.',
      title: 'Where the rate comes from', lines: [
        { tex: `\\partial_t p_t(\\cdot\\mid\\x_1) = \\dot\\kappa_t\\,(\\delta_{x_1^i} - p_0) = ${CHG('\\frac{\\dot\\kappa_t}{1 - \\kappa_t}\\big(\\delta_{x_1^i} - p_t(\\cdot\\mid\\x_1)\\big)')}`, why: 'Differentiate the path, then use \\(\\delta - p_t = (1 - \\kappa_t)(\\delta - p_0)\\).', uses: ['expectation'] },
        { tex: `\\text{conditional rate to } y = ${CHG('\\frac{\\dot\\kappa_t}{1 - \\kappa_t}\\,\\mathbf 1\\{y = x_1^i\\}')}`, why: 'A chain that jumps to the clean token at this rate, from wherever it is, changes the distribution by exactly the right-hand side.', uses: ['continuity'] },
        { tex: `\\text{marginal rate to } y = \\frac{\\dot\\kappa_t}{1 - \\kappa_t}\\sum_{\\x_1}p(\\x_1\\mid\\x_t)\\,\\mathbf 1\\{y = x_1^i\\} = ${CHG('\\frac{\\dot\\kappa_t}{1 - \\kappa_t}P(x_1^i = y\\mid\\x_t)')}`, why: 'Average the conditional rate over the posterior of the clean sentence given the current state.', uses: ['cond-exp'] },
      ],
    },
    after: ['A token of the masked chain that is already visible equals its clean value, so its posterior marginal is a point mass on itself and the chain never revises it. A token of the uniform chain may be noise, and the posterior keeps re-weighing it against the rest of the sentence, so the chain revises tokens it has placed. At N = 1 both chains draw every token independently from its marginal in one step, and almost no sentence is valid. Both need many steps for the posterior to couple the tokens, and the readout shows how many.'],
    check: {
      q: 'Why does the masked chain never revise a token it has revealed?',
      opts: [['A revealed token equals its clean value, so its posterior marginal is a point mass on itself', true], ['The rate \\(\\dot\\kappa_t/(1 - \\kappa_t)\\) is zero for revealed tokens', false], ['The Euler step skips revealed tokens to save computation', false]],
      fb: { true: 'Right. Under the masked source a visible token can only be the clean one, so \\(P(x_1^i = y\\mid\\x_t)\\) is zero for every other y, and the rate of a change is zero. Under the uniform source a visible token may be noise, and the posterior can move it.', false: 'The rate factor is the same for every token. What differs is the posterior: a visible token of the masked chain is known to be clean, so the posterior marginal puts all its mass on the current value.' },
    },
  },
  {
    id: 'parallel', kicker: 'Speed against coherence', title: 'Parallel decoding and its price', scene: 'parallel',
    prose: [
      'The appeal of masked diffusion for language is that a sampler can reveal several tokens per network evaluation. It reveals them independently, however: two tokens unmasked in the same step are drawn from the product of their conditionals instead of from their joint conditional. When the two are coupled, as an animal and its sound are, the product produces combinations the language never contains. Confidence-based samplers from MaskGIT reveal only tokens whose predicted probability exceeds a threshold τ.',
      'The toy language is decoded with one of two rules: k random masked tokens per step, or every token whose drawn value has probability above τ, with at least one per step. Each point of the plot averages 1000 seeded runs, and the current setting is highlighted. The right panel compares the entropy of the produced sentences with the entropy of the language. Switch the rule and move k or τ.',
    ],
    tex: `\\mathrm{KL}\\Big(p(x^i, x^j\\mid${xt})\\,\\Big\\|\\,p(x^i\\mid${xt})\\,p(x^j\\mid${xt})\\Big) = ${I}(x^i; x^j\\mid${xt})`,
    deriv: {
      have: ['Two masked tokens i and j are revealed in one step, each from its conditional given the visible tokens \\(\\x_t\\).'],
      goal: '\\text{error of the step} = I(x^i; x^j\\mid\\x_t)',
      plan: 'Write the KL divergence from the joint conditional to the product of the two conditionals and recognize the mutual information.',
      title: 'Why the error is the mutual information', lines: [
        { tex: `\\mathrm{KL} = \\sum_{a, b}p(a, b\\mid\\x_t)\\log\\frac{p(a, b\\mid\\x_t)}{${CHG('p(a\\mid\\x_t)\\,p(b\\mid\\x_t)')}}`, why: 'The definition of the KL divergence, with the product of the conditionals in the denominator.', uses: ['kl'] },
        { tex: `\\mathrm{KL} = ${CHG('H(x^i\\mid\\x_t) + H(x^j\\mid\\x_t) - H(x^i, x^j\\mid\\x_t)')} = I(x^i; x^j\\mid\\x_t)`, why: 'Split the logarithm into three sums. Each is an entropy given the visible tokens, and their combination is the conditional mutual information.', uses: ['log-rules'] },
        { tex: `\\text{animal and sound, nothing visible:}\\quad I = H(\\text{animal}) = ${CHG('1.85\\text{ bits}')}`, why: 'The sound is a function of the animal, so the joint entropy equals the entropy of one of them, and the mutual information is the full entropy of the weights 0.4, 0.3, 0.2 and 0.1.', uses: ['log-rules'] },
      ],
    },
    after: ['Revealing one token per step is exact and takes eight steps. Revealing all eight at once takes one step and produces a valid sentence in about 3% of the runs. The threshold rule stays coherent in about three steps at τ = 0.35, but it reveals mostly the most probable values, so the entropy of its output falls far below that of the language. The same trade between steps, coherence and diversity reappears in the samplers of §2.10 and as the effect of guidance in §2.11.'],
    check: {
      q: 'The animal and its sound are both masked and are revealed independently in the same step, each from the weights 0.4, 0.3, 0.2 and 0.1. With what probability do they match?',
      answer: 0.3, tol: 0.005,
      fb: { true: 'Right. They match when both draws pick the same animal, which has probability \\(\\sum_i w_i^2 = 0.16 + 0.09 + 0.04 + 0.01 = 0.30\\).', false: 'Both draws must pick the same animal. The probability is \\(\\sum_i w_i^2 = 0.4^2 + 0.3^2 + 0.2^2 + 0.1^2\\).' },
    },
    closing: `<h3>Where this leads</h3>
      <p>Every piece of the continuous theory has a discrete counterpart: the Kolmogorov equation for the Fokker–Planck equation, the concrete score for the score, score entropy for denoising score matching, and the masked ELBO for the variational bound of §2.6. What is new is the cost of parallel generation, which is a property of the factorization and not of the network. <a href="../hybrid/">§2.10</a> places masked diffusion, autoregressive models and next-scale prediction on one axis of factorizations, and §2.11 turns to guidance, which concentrates samples the way the confidence threshold does here.</p>
      <p>The earlier tutorials <a href="../sde/">Stochastic differential equations</a>, <a href="../score-matching/">Denoising and score matching</a> and <a href="../likelihood/">Likelihood and the ELBO</a> develop the continuous counterparts used on this page.</p>`,
  },
];
