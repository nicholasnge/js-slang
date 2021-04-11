import { GPU } from 'gpu.js'
import { parse } from 'acorn'
import { generate } from 'astring'
import * as es from 'estree'
import { gpuRuntimeTranspile } from './transfomer'
import { ACORN_PARSE_OPTIONS } from '../constants'
import { TypeError } from '../utils/rttc'

const pr = (pre: string, string: string | undefined): void => {
  process.stdout.write(pre + ': ' + string + '\n')
}

// Heuristic : Only use GPU if array is bigger than this
const MAX_SIZE = 200

// helper function to build 1-3 D array output
function copyArray(res: any, arr: any, end: number[]) {
  if (end.length === 1) {
    for (let i = 0; i < end[0]; i++) {
      arr[i] = res[i]
    }
    return;
  }
  if (end.length === 2) {
    for (let i = 0; i < end[0]; i++) {
      arr[i] = []
      for (let j = 0; j < end[1]; j++) {
        arr[i][j] = res[i][j]
      }
    }
    return;
  }
  if (end.length === 3) {
    for (let i = 0; i < end[0]; i++) {
      arr[i] = []
      for (let j = 0; j < end[1]; j++) {
        arr[i][j] = []
        for (let k = 0; k < end[2]; k++) {
          arr[i][j][k] = res[i][j][k]
        }
      }
    }
    return;
  }
}

// helper function to check array is initialized
function checkArray(arr: any): boolean {
  return Array.isArray(arr)
}


/*
 * we only use the gpu if:
 * 1. we are working with numbers
 * 2. we have a large array (> 100 elements)
 */
function checkValidGPU(f: any, end: any): boolean {
  let res: any
  if (end.length === 1) res = f(0)
  if (end.length === 2) res = f(0, 0)
  if (end.length === 3) res = f(0, 0, 0)

  // we do not allow array assignment
  // we expect the programmer break it down for us
  if (typeof res !== 'number') {
    return false
  }

  let cnt = 1
  for (const i of end) {
    cnt = cnt * i
  }

  return cnt > MAX_SIZE
}

// just run on js!
function manualRun(f: any, end: any, res: any) {
  function build() {
    for (let i = 0; i < end[0]; i++) {
      res[i] = f(i)
    }
    return
  }

  function build2D() {
    for (let i = 0; i < end[0]; i = i + 1) {
      for (let j = 0; j < end[1]; j = j + 1) {
        res[i][j] = f(i, j)
      }
    }
    return
  }

  function build3D() {
    for (let i = 0; i < end[0]; i = i + 1) {
      for (let j = 0; j < end[1]; j = j + 1) {
        for (let k = 0; k < end[2]; k = k + 1) {
          res[i][j][k] = f(i, j, k)
        }
      }
    }
    return
  }

  if (end.length === 1) return build()
  if (end.length === 2) return build2D()
  return build3D()
}

/* main function that runs code on the GPU (using gpu.js library)
 * @end : end bounds for array
 * @extern : external variable definitions {}
 * @f : function run as on GPU threads
 * @arr : array to be written to
 */
export function __createKernel(end: any, extern: any, externFn: any[], f: any, arr: any, f2: any) {
  const gpu = new GPU()

  // check array is initialized properly
  let ok = checkArray(arr)
  let err = ''
  if (!ok) {
    err = typeof arr
  }

  // // TODO: find a cleaner way to do this
  // if (end.length > 1) {
  //   ok = ok && checkArray2D(arr, end)
  //   if (!ok) {
  //     err = 'undefined'
  //   }
  // }
  //
  // if (end.length > 2) {
  //   ok = ok && checkArray3D(arr, end)
  //   if (!ok) {
  //     err = 'undefined'
  //   }
  // }
  // err;
  //
  // if (!ok) {
  //   throw new TypeError(arr, '', 'object or array', err)
  // }
  if (!Array.isArray(arr)) {
    throw new TypeError(arr, '', 'object or array', err)
  }

  // check if program is valid to run on GPU
  ok = checkValidGPU(f2, end)
  if (!ok) {
    manualRun(f2, end, arr)
    return
  }

  const nend = []
  for (let i = end.length - 1; i >= 0; i--) {
    nend.push(end[i])
  }

  // external variables to be in the GPU
  const out = { constants: {} }
  out.constants = extern
  externFn.forEach((x)=>gpu.addFunction(x))
  const gpuFunction = gpu.createKernel(f, out).setOutput(nend)
  pr("added fn", "OK")
  const res = gpuFunction() as any
  pr("ran fn", "OK")

  // Output from gpu is Float32Array[] for 2 dim and Float32Array[][] for 3 dim
  // if we copy the output to our target array, we incur O(n^<dimensions>) cost
  // if we dont copy over: we have to be content with the FloatArray32 and set
  // any references to the target array to our new output array

  // uncomment to incur O(n^<dimensions>) for full "correctness"
  // if (end.length === 2) res = res.map((x : any) => Array.from(x));
  // if (end.length === 3) res = res.map((x : any) => Array.from(x));
  copyArray(res, arr, end)
  return res
}

function entriesToObject(entries: [string, any][]): any {
  const res = {}
  entries.forEach(([key, value]) => {res[key] = value})
  return res
}

/* tslint:disable-next-line:ban-types */
const kernels: Map<number, Function> = new Map()

export function __clearKernelCache() {
  kernels.clear()
}

export function __createKernelSource(
  end: number[],
  externSource: [string, any][],
  localNames: string[],
  arr: any,
  f: any,
  kernelId: number
) {
  const extern = entriesToObject(externSource)
  const externFn = [];
  for (const prop in extern) {
    if (typeof extern[prop] === "function") {
      externFn.push(extern[prop])
      delete extern[prop]
    }
  }

  const memoizedf = kernels.get(kernelId)
  if (memoizedf !== undefined) {
    return __createKernel(end, extern, externFn, memoizedf, arr, f)
  }

  const code = f.toString()
  // We don't need the full source parser here because it's already validated at transpile time.
  const ast = (parse(code, ACORN_PARSE_OPTIONS) as unknown) as es.Program
  const body = (ast.body[0] as es.ExpressionStatement).expression as es.ArrowFunctionExpression
  const newBody = gpuRuntimeTranspile(body, new Set(extern.keys()), new Set(localNames))
  const kernel = new Function(generate(newBody))
  kernels.set(kernelId, kernel)
  pr("gpuRuntimeTranspile", "OK")

  return __createKernel(end, extern, externFn, kernel, arr, f)
}
