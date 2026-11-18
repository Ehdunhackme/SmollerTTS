import {toWords,toOrdinal} from 'to-words/en-US';
const options={casing:'lower',decimalStyle:'digit-by-digit'};
const number=x=>toWords(String(x),options);

export function normalizeEnglish(input){
  let text=input.normalize('NFKC').replace(/[’‘]/g,"'");
  if(/[^\x00-\x7f]/.test(text.replace(/[—–…“”]/g,'')))
    text=text.normalize('NFD').replace(/\p{M}/gu,'');
  if(/[^\x00-\x7f—–…“”]/.test(text))throw new Error('This model supports English text. Please spell unfamiliar names with English letters.');
  text=text.replace(/&/g,' and ');
  text=text.replace(/\b(Dr|Mr|Mrs|Ms|Prof)\./g,(_,x)=>({Dr:'doctor',Mr:'mister',Mrs:'misses',Ms:'miss',Prof:'professor'})[x]);
  text=text.replace(/\b(?:CPU|USB|PDF|AI|GPU|API|URL|HTTP)\b/g,x=>x.split('').join(' '));
  text=text.replace(/\$(\d[\d,]*)(?:\.(\d{1,2}))?/g,(_,a,b)=>{
    const dollars=Number(a.replaceAll(',','')),cents=b?Number(b.padEnd(2,'0')):0;
    return `${number(dollars)} ${dollars===1?'dollar':'dollars'}${cents?` and ${number(cents)} ${cents===1?'cent':'cents'}`:''}`;
  });
  if(/\d\s*[:/]\s*\d|[£€¥]\s*\d/.test(text))throw new Error('Please spell out times, dates and non-dollar currency amounts.');
  text=text.replace(/\b(\d+)(?:st|nd|rd|th)\b/gi,(_,x)=>toOrdinal(x,options));
  text=text.replace(/[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?/g,x=>{
    const percent=x.trim().endsWith('%'),digits=x.replace(/[\s,%]/g,'');
    if(!Number.isFinite(Number(digits))||Math.abs(Number(digits))>1e12)throw new Error('Please spell out very large numbers.');
    return number(digits)+(percent?' percent':'');
  });
  if(/[$£€¥\d]/.test(text))throw new Error('This number format needs to be written out.');
  return text.replace(/[—–]/g,',').replace(/…/g,'.');
}

export function pronounceEnglish(text,lexicon){
  const normalizedText=normalizeEnglish(text),chunks=[];
  let tokens=[1],stress=[0],words=[];const missing=[];
  const flush=()=>{
    if(words.length){if(tokens.at(-1)!==1){tokens.push(1);stress.push(0);}chunks.push({tokens,stress,words});}
    tokens=[1];stress=[0];words=[];
  };
  for(const item of normalizedText.toLowerCase().match(/[a-z]+(?:'[a-z]+)*|[,.;:!?]/g)||[]){
    if(/^[,.;:!?]$/.test(item)){
      if(tokens.at(-1)!==1){tokens.push(1);stress.push(0);}
      if(/[.!?]/.test(item))flush();
      continue;
    }
    const phones=lexicon.get(item);
    if(!phones){missing.push(item);continue;}
    if(tokens.length+phones.length>190)flush();
    if(phones.length>180)throw new Error('One word is too long for this voice.');
    for(const [phone,,accent] of phones){tokens.push(phone);stress.push(accent);}
    words.push(item);
  }
  flush();
  if(missing.length)throw new Error(`Pronunciation is missing for: ${[...new Set(missing)].join(', ')}. No words were skipped.`);
  if(!chunks.length)throw new Error('Enter an English sentence.');
  return {normalizedText,chunks};
}
