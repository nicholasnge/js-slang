import { generate } from 'astring'
import { mockContext } from '../../mocks/context'
import { parse } from '../../parser/parser'
import { stripIndent } from '../../utils/formatters'
import { transpileToGPU } from '../../gpu/gpu'
// import { VariableDeclaration, VariableDeclarator } from 'estree'
import { __createKernel, __createKernelSource } from '../lib'

test('simple for loop gets transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
        res[i] = i;
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  pr("", transpiled)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('many simple for loop gets transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
        res[i] = i;
    }

    let res1 = [];
    for (let i = 0; i < 5; i = i + 1) {
      res1[i] = i;
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(2)
})
const pr = (pre: string, string: string | undefined): void => {
  process.stdout.write(pre + ': ' + string + '\n')
}

test('simple for loop with constant condition transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    const c = 10;
    let i = 0;
    for (i = 0; i < c; i = i + 1) {
        res[i] = i;
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  // pr('program type', program.type)
  // pr('program body', program.body.toString())
  //
  // pr('variable declaration', program.body[0].type) // variable declaration
  // pr('vardec kind: ', (program.body[0] as VariableDeclaration).kind) // let: let/const
  // pr('vardec declarator: ', (program.body[0] as VariableDeclaration).declarations[0].type) // variable declarator
  // pr('vardec declarator:identifier: ', (program.body[0] as VariableDeclaration).declarations[0].id.type) // identifier
  // pr('vardec declarator:expr: ', ((program.body[0] as VariableDeclaration).declarations[0] as VariableDeclarator).init?.type.toString()) // expr: ArrayExpression
  //
  // pr('variable declaration 2:', program.body[1].type) // variable declaration
  // pr('vardec kind: ', (program.body[1] as VariableDeclaration).kind) // let: let/const
  // pr('vardec declarator: ', (program.body[1] as VariableDeclaration).declarations[0].type) // variable declarator
  // pr('vardec declarator:identifier: ', (program.body[1] as VariableDeclaration).declarations[0].id.type) // identifier
  // pr('vardec declarator:expr: ', ((program.body[1] as VariableDeclaration).declarations[0] as VariableDeclarator).init?.type.toString()) // expr: ArrayExpression
  //
  // pr('program body 1: ', program.body[1].type)
  // pr('program body 2: ', program.body[2].type)

  transpileToGPU(program)

  const transpiled = generate(program)
  // pr('program type', program.type)
  // pr('program body', program.body.toString())
  // pr('variable declaration', program.body[0].type) // expr statement
  // pr('variable declaration', program.body[1].type) // expr statement
  // pr('variable declaration', program.body[2].type) // variable declaration
  // pr('variable declaration', program.body[3].type) // variable declaration
  // pr('variable declaration', program.body[4].type) // expr statement
  console.log('T: ' + transpiled.toString())

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('simple for loop with let condition transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    let c = 10;
    for (let i = 0; i < c; i = i + 1) {
        res[i] = i;
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  pr("T", transpiled)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('simple for loop with math function call transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    let c = 10;
    for (let i = 0; i < c; i = i + 1) {
        res[i] = math_abs(i);
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  pr("T", transpiled)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('simple for loop with different end condition transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    const f = () => 5;
    let c = f();
    for (let i = 0; i < c; i = i + 1) {
        res[i] = i;
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  pr("T", transpiled)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

//TODO FAILING
test('2 for loop case gets transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 15; i = i + 1) {
        res[i] = [];
        for (let j = 0; j < 15; j = j + 1) {
            res[i][j] = i * j;
            res[i][j] = res[i][j] + 1;
        }
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  console.log("\n\n\n" + transpiled.toString())
  console.log("\n\n\n" + "display(\"Attempting to optimize 2 levels of nested loops starting on line 2\");\n" +
    "__clearKernelCache();\n" +
    "let res = [];\n" +
    "__createKernelSource([15, 15], [], [\"temp\"], res, (i, j) => {\n" +
    "  return i * j;\n" +
    "}, 0);")
  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('new test lambda functions passed in', () => {
  const code = stripIndent`
    let res = [];
    const fn = (x,y) => x + y;
    for (let i = 0; i < 15; i = i + 1) {
        res[i] = [];
        for (let j = 0; j < 15; j = j + 1) {
            res[i][j] = fn(i, j);
        }
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  console.log('T: ' + transpiled.toString())
  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('2 for loop case with body gets transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
        res[i] = 0;
        for (let j = 0; j < 5; j = j + 1) {
            res[i] = res[i] + j;
        }
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  console.log('T: ' + transpiled.toString())


  const cnt = transpiled.match(/__createKernelSource/g)?.length
  pr('2 for loop case', transpiled)
  expect(cnt).toEqual(1)
})

test('2 for loop case with 2 indices being written to gets transpiled correctly', () => {
  const code = stripIndent`
let res1 = [];for (let i = 0; i < 201; i = i + 1) {for (let j = 0; j < 201; j = j + 1) {res1[i][j] = j;}}
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  pr('T: ' + transpiled.toString(), "")

  __createKernelSource([201, 201], [], [], [], (i: number, j: number) => {
    return j;
  }, 0);
  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('2 for loop case with 2 indices being written + local updates to gets transpiled correctly', () => {
  const code = stripIndent`
let res1 = [];
for (let i = 0; i < 201; i = i + 1) {
  res1[i] = 0;
  for (let j = 0; j < 201; j = j + 1) {
    res1[i] = res1[i] + j;
  }
}
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  pr('T: ' + transpiled.toString(), "")

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('2 for loop case with 2 indices being written + use of result variable[i][j] gets transpiled', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
      res[i] = [];
      for (let j = 0; j < 5; j = j + 1) {
        res[i][j] = j;
      }
    }

    for (let i = 0; i < 5; i = i + 1) {
        for (let j = 0; j < 5; j = j + 1) {
            let x = res[i][j];
            let y = math_abs(x * -5);
            res[i][j] = x + y;
        }
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)
  pr(transpiled.toString(), "")

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('3 for loop case with 1 index being written to gets transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
        for (let j = 0; j < 5; j = j + 1) {
            for (let k = 0; k < 5; k = k + 1) {
              res[i] = i*j;
            }
        }
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('3 for loop case with 2 indices being written to gets transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
        for (let j = 0; j < 5; j = j + 1) {
            for (let k = 0; k < 5; k = k + 1) {
              res[i][j] = i*j;
            }
        }
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('3 for loop case with 3 indices being written to gets transpiled correctly', () => {
  const code = stripIndent`
    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
        for (let j = 0; j < 5; j = j + 1) {
            for (let k = 0; k < 5; k = k + 1) {
              res[i][j][k] = i*j;
            }
        }
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1)
})

test('many for loop case - matrix multiplication (2 transpilations)', () => {
  const code = stripIndent`
    const size = 10;
    const L = [];
    const R = [];
    for (let r = 0; r < size; r = r + 1) {
        L[r] = [];
        R[r] = [];
        for (let c = 0; c < size; c = c + 1) {
            L[r][c] = r*c;
            R[r][c] = r + c;
        }
    }

    const res = [];
    for (let r = 0; r < size; r = r + 1) {
        res[r] = [];
    }

    for (let r = 0; r < size; r = r + 1) {
        for (let c = 0; c < size; c = c + 1) {
            let sum = 0;
            for (let i = 0; i < size; i = i + 1) {
                sum = sum + L[r][i] * R[i][c];
            }
            res[r][c] = sum;
        }
    }
  `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)

  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(2)
})

test('resolve naming conflicts by disabling automatic optimizations', () => {
  const code = stripIndent`
    const __createKernelSource = 10;

    let res = [];
    for (let i = 0; i < 5; i = i + 1) {
        res[i] = i;
    }
    `
  const context = mockContext(4, 'gpu')
  const program = parse(code, context)!
  transpileToGPU(program)
  const transpiled = generate(program)

  // a new kernel function with name __createKernelSource0 should be created here
  const cnt = transpiled.match(/__createKernelSource/g)?.length
  expect(cnt).toEqual(1) // Occurrence comes from the warning below
  const cntWarnings = transpiled.match(
    /display\("Manual use of GPU library symbols detected, turning off automatic GPU optimizations."\)/g
  )?.length
  expect(cntWarnings).toEqual(1)
})
