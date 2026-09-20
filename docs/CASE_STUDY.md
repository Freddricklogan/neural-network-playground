# Case Study — Neural Network Playground

**Repository:** [neural-network-playground](https://github.com/Freddricklogan/neural-network-playground) · **Live demo:** [freddricklogan.github.io/neural-network-playground](https://freddricklogan.github.io/neural-network-playground/) · **Author:** Freddrick Logan

---

## 1. Who has this problem

An instructor teaching machine learning to people who will use it rather than research it — a business-school analytics course, a bootcamp cohort, a workshop for analysts in a state agency. The learners can call `model.fit()` on day one. What they cannot do is explain why the learning rate they were given is right, what a decision boundary is, or why their model memorised the training data. The framework did the mechanism, so it was never learned.

## 2. The problem, as a scenario

Week three of an applied ML module. A student has trained a classifier in a notebook, the accuracy is high, and the instructor asks what regularisation did to the weights. The student changes the parameter and reruns; a number moves; nobody can say what happened between the two numbers. The instructor draws a decision boundary on the whiteboard and describes gradient descent with hand gestures. Half the room follows; the other half will carry a black-box understanding into jobs where they must justify a model to someone who understands even less.

## 3. What it costs to leave it alone

Analysts who tune by superstition. Graduates who cannot explain a model cannot defend it to a compliance reviewer, cannot recognise overfitting, and cannot tell a stakeholder what would change the outcome. I will not attach a figure to that; the remedy — intuition for the mechanism before the abstraction — is what this project is for. The nearer cost is pedagogical: a lesson built on `model.fit()` produces the same confusion in every cohort, every term.

## 4. The approach, and the alternative I rejected

I wrote a small neural network from scratch — matrices, activation functions, softmax cross-entropy, L1 and L2 regularisation, mini-batch gradient descent — with no machine-learning framework, running entirely in the browser. Every hyperparameter is a live control and the decision boundary redraws as the network trains, so cause and effect appear in the same frame. Runs are seeded: an instructor hands out a seed and every student reproduces the identical run, turning "what did you see?" into a conversation about one shared artefact.

The alternative was to wrap TensorFlow.js. It would have trained faster and supported bigger models, and it would have hidden exactly the mechanism the lesson is about. An engine small enough to read in an hour is the point; being able to unit-test the mathematics directly is what made the rest of this project possible.

## 5. What the code does today

Real: the matrix library, the activation functions and their derivatives, the forward and backward passes, softmax cross-entropy, L1 and L2 regularisation, mini-batch training with a seeded generator, six synthetic dataset generators, accuracy and loss metrics, and the decision-boundary and loss-curve rendering. The engine is pure logic separated from the DOM; the tests exercise it directly.

Simulated: the data. All six datasets are synthetic — XOR, concentric circles, a spiral, Gaussian clusters, two moons and a sine wave — generated from a seed, at a scale small enough to watch train. The page says so.

Worth knowing: rebuilding this exposed that the published demo did not run at all — the first forward pass threw on a matrix-shape check because the bias vector could not be added to a batch. The reported loss was also computed against the wrong target matrix, the regularisation penalty was applied without the learning rate, and mini-batches were sampled with replacement. All are fixed and tested; the full list is in the audit file.

## 6. Evidence

Measured in continuous integration on the current main branch: 81 unit tests passing across six files, 100% statement coverage over the engine, lint and HTML validation clean, CodeQL and dependency scanning enabled. The tests assert that loss falls on a linearly separable problem, that XOR is learned above chance, that L1 and L2 shrink weights relative to no regularisation, that the penalty scales with the learning rate, and that a given seed reproduces an identical network. Headless-browser smoke test: zero console errors; on the default seed, training reached epoch 56 at 99.67% accuracy. Security posture: Content Security Policy with `default-src 'none'`, the one CDN library pinned with a Subresource Integrity hash and vendored as a fallback.

## 7. What it would take to run this in production

As a classroom tool it is already deployable: static files, no accounts, no backend. Inside a course at scale it would need an LTI 1.3 launch so it opens from the learning management system with the student's identity; a way for the instructor to publish a seed and parameter set as an assignment; capture of runs as xAPI statements to a learning-record store; and export of trained weights so a later notebook exercise can start from them. That is weeks of integration work, none of it touching the engine. Hosting stays static; the record store is the only new service.

## 8. Limits and next steps

Dense layers only; no convolution, dropout, learning-rate schedules or train/validation split; a scale capped at what a browser tab can draw. Next: training in a Web Worker so the interface stays responsive on larger runs, a held-out set so overfitting is visible as a diverging curve rather than an assertion, and a finite-difference gradient check in the test suite.

## 9. Who should look at this

**Hiring manager:** evidence that I can implement the mathematics rather than call it, and that I found and fixed a demo that had never actually run.
**Consulting client:** a teaching instrument for an applied-ML curriculum, ready to drop into a course with seeded, reproducible exercises.
**Engineer:** read `src/network.js` and `src/matrix.js` for the engine and `tests/network.test.js` for what the training claims are checked against; the audit file explains the defects that hid behind a working-looking page.
