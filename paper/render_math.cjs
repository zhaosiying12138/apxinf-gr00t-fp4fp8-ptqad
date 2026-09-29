// KaTeX 0.16.22 is vendored with its MIT license; no network at build/read time.
const fs = require('fs');
const katex = require('./assets/katex/katex.min.js');
const items = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(items.map(({tex, display}) =>
  katex.renderToString(tex, {displayMode: display, throwOnError: true,
    output: 'htmlAndMathml', strict: 'ignore', trust: false}))));
