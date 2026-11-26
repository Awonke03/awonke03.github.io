import assert from 'node:assert/strict';
import fs from 'node:fs';
import {vector,scores,csvParse,quote,splitClauses} from '../sentiscope/engine.mjs';
const fixtures=JSON.parse(fs.readFileSync(new URL('./parity.json',import.meta.url)));
for(const {text,vector:v} of fixtures){const actual=vector(text);assert.equal(actual.size,Object.keys(v).length);for(const [k,x] of Object.entries(v))assert.ok(Math.abs(actual.get(Number(k))-x)<1e-7)}
const m=JSON.parse(fs.readFileSync(new URL('../sentiscope/models.json',import.meta.url)));
for(const model of Object.values(m.sentiment)){assert.equal(model.classes.length,3);const x=scores(model,'The service is excellent');assert.ok(Math.abs(x.reduce((s,a)=>s+a.score,0)-1)<1e-6)}
assert.equal(m.emotion.classes.length,28);
assert.deepEqual(csvParse('text,language\n"hello, world",en\n"a""b",ts'),[['text','language'],['hello, world','en'],['a"b','ts']]);
assert.ok(quote('=2+2').startsWith('"\''));
assert.equal(splitClauses('The app is great but fees are high').length,2);
console.log('Portable model parity, inference and CSV checks passed.');

import {newsUrl,newsRows,safeUrl,explainWords,searchNews} from '../sentiscope/web-intelligence.mjs';
assert.equal(new URL(newsUrl('Vodacom',7)).searchParams.get('query'),'"Vodacom"');
assert.equal(new URL(newsUrl('Vodacom',7)).searchParams.get('timespan'),'7d');
assert.equal(safeUrl('javascript:alert(1)'),'');
assert.equal(newsRows({articles:[{url:'https://example.org/a',title:'Actual headline',language:'French'},{url:'https://example.org/b',title:'Actual headline'},{url:'javascript:alert(1)',title:'Bad'}]}).length,1);
const fake=async text=>[{label:'positive',score:text.includes('excellent')?.9:.2}];
const e=await explainWords('excellent service','positive',fake);assert.ok(Math.abs(e.items[0].delta-.7)<1e-9);assert.equal(e.items[1].delta,0);
assert.equal((await searchNews('Vodacom',30,async()=>({ok:true,text:async()=>JSON.stringify({articles:[{url:'https://example.org/a',title:'Headline'}]})}))).length,1);
console.log('News parsing, URL safety and perturbation attribution tests passed.');
