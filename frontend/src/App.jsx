import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Upload, Play, Check, Loader2, AlertCircle, Download, Lock } from 'lucide-react';


/* =====================================================================

   Photoencephalography (PEG) analysis, step by step

   Mirrors "Data analysis" of Bonzanni, Rouleau & Murugan, STAR Protocols 6, 104223 (2025).



   BACKEND CONTRACT (to implement next in FastAPI)

   POST /sessions                       multipart files[]            -> {session_id, fields[], n_subjects, n_points, preview:{field:number[]}, tasks?:string[]}

   POST /sessions/{sid}/stats_dataset   multipart file               -> {columns[], n_rows}

   POST /sessions/{sid}/steps/{id}      JSON {action, params}        -> result object (shapes below), merged into the step's result

   Every result may carry: downloads:[{label,url}] (url relative to API_URL), warnings:[string]



   clustering : {z:number[][], row_labels[], dendrogram:{icoord,dcoord}, clusters:number[], silhouette:{k[],score[]}}   (rows already in dendrogram order)

   descriptives: {average:{field:number[]}, cv:{field:number[]}}

   entropy    : {entropy:{field:number[]}, bin_width:number, fd_widths:{field:number}}

   stft       : run -> {trace_names[], shape:[n,m], avg_power:{name,f[],p[]}}   compare -> {compare:[{label,f[],t[],z[][],avg_f[],avg_p[]}]}

   plot_spec  : {plots:[{name,f[],t[],z[][]}]}

   dfa        : filter -> {filter:{f[],mag_db[]}}

                single -> {white:{mean,sd,n,ok,slopes:{x[],y[]}}, trace:{exponent,x_all[],y_all[],x_fit[],y_fit[],line_y[],signal[],envelope[]}}

                loop   -> {loop:{cols[],values:number[][]}}   (rows = subjects)

   correlation | coherence | granger | mutual_info:

                {pairs[], median:{pair:number[]}, trajectories:{pair:number[][]} (pair -> subject -> window),

                 mean_network:{labels[],z[][]}, trgc?:{pair:{index:number[], fraction_above:number}}}

   stats      : fit -> {singular:boolean, summary:string, anova:object[], posthoc:object[]}   compare -> {compare:object[]}

   ===================================================================== */



const API_URL = 'http://localhost:8000';

const api = axios.create({ baseURL: API_URL });

const ACCENT = '#3347c9';



/* ----------------------------- step definitions ----------------------------- */

const NET_PARAMS = [

  { k: 'numSubjects', label: 'numSubjects', type: 'number', def: 20, help: 'Number of subjects analyzed.' },

  { k: 'resampleFactor', label: 'ResampleFactor', type: 'number', def: 1, help: 'If 1, no resampling.' },

  { k: 'cutoff', label: 'cutoff', type: 'number', def: 0, help: 'Threshold for eliminating weak edges.' },

  { k: 'fs', label: 'fs (Hz)', type: 'number', def: 20, help: 'Sampling frequency.' },

  { k: 'timeWindow', label: 'TimeWindow (s)', type: 'number', def: 60, help: 'Length of each time window.' },

  { k: 'overlap', label: 'Overlap', type: 'number', def: 0.5, help: 'Fraction of overlap between consecutive windows.' },

];

const NET_NOTE = 'Median across windows compresses time and treats a dynamic network as static; the time-resolved values are provided as well.';



const STEPS = [

  { id: 'about', group: 'Start', title: 'Overview', kind: 'about' },

  { id: 'dataset', group: 'Start', no: 1, title: 'Create dataset', script: 'OrganizeUPEsExcelFile / UPEs.mat', time: '5–10 min', kind: 'dataset',

    aim: 'Raw UPE traces are organized in one structure, UPEs. Each field is an electrode (Background, Occipital, Temporal) or an electrode-task pair. Each field holds rows = time points and columns = subjects. All later steps use it.',

    notes: ['Excel input: one sheet per electrode (or electrode-task pair), columns are subjects with the subject ID in row 1, rows are time points.',

      'If the trace needs filtering, filter the entire trace first; some filters add time delays.'] },

  { id: 'clustering', group: 'Self-properties of UPEs', no: 2, title: 'Hierarchical clustering', script: 'HierarchicalClusteringUPEs.m', time: '5–10 min', kind: 'clustering',

    aim: 'Hierarchical clustering of the raw UPE traces to assess temporal similarity across conditions and participants, and to identify recurring patterns in how the signals change over time.',

    question: 'Which traces (subject-electrode pairs) share similar temporal patterns?',

    assumptions: ['Signal stationarity.'],

    pitfalls: ['Clustering may be driven by artifacts or noise.', 'Where to cut the dendrogram is arbitrary.', 'Standardizing removes amplitude information.', 'Highly nonlinear relationships are not captured.'],

    notes: ['Look for large gaps in branch heights, or use a fixed number of clusters (for example the number of electrodes). Silhouette scores help guide the choice.'],

    params: [

      { k: 'numSubjects', label: 'numSubjects', type: 'number', def: 20, help: 'Number of subjects analyzed.' },

      { k: 'cluster', label: "'Cluster'", type: 'select', def: 'Column', opts: ['Column', 'Row'], help: "'Column' clusters along time and reorders the rows (subject-electrode pairs)." },

      { k: 'standardize', label: "'Standardize'", type: 'select', def: 'Row', opts: ['Row', 'Column', 'None'], help: "'Row' gives each trace mean 0 and std 1, so patterns matter more than absolute values." },

      { k: 'method', label: 'linkage', type: 'select', def: 'average', opts: ['average', 'complete', 'single', 'ward'], help: 'Linkage method.' },

      { k: 'metric', label: 'distance', type: 'select', def: 'euclidean', opts: ['euclidean', 'correlation', 'cityblock'], help: 'Distance metric.' },

      { k: 'nClusters', label: 'clusters to cut', type: 'number', def: 3, help: 'Used for the cluster labels and silhouette score.' },

    ] },

  { id: 'descriptives', group: 'Self-properties of UPEs', no: 3, title: 'Descriptive statistics', script: 'DescriptiveStatisticsUPEs.m', time: '1–2 min', kind: 'box',

    aim: 'Average UPE count and coefficient of variation (CV) for every trace.',

    question: 'How intense is each trace on average, and how variable is it (CV as proxy of variability)?',

    notes: ['Alternatives to CV: interquartile range, root mean square of successive differences, signal-to-noise ratio.'],

    params: [{ k: 'numSubjects', label: 'numSubjects', type: 'number', def: 20, help: 'Number of subjects analyzed.' }],

    boxes: [['average', 'Average UPE count'], ['cv', 'Coefficient of variation']] },

  { id: 'entropy', group: 'Self-properties of UPEs', no: 4, title: 'Shannon entropy', script: 'ShannonEntropyUPEs.m', time: '2–5 min', kind: 'box',

    aim: 'Entropy of every trace as a proxy of signal complexity or irregularity. A Freedman-Diaconis bin width is computed per field, the median across fields sets one common bin-edge list, and entropy is computed on it.',

    question: 'What is the level of irregularity or complexity in the signal?',

    assumptions: ['Stationarity (analyze shorter windows if needed).', 'Enough dynamic range to discretize into bins.', 'A common binning scheme assumes comparable distributions across signals.'],

    pitfalls: ['Sensitive to noise, which can look like complexity.', 'The binning strategy influences the result.', 'Short intervals give unstable estimates.'],

    notes: ['Alternatives: sample, approximate and permutation entropy. Run separately for each task.'],

    params: [{ k: 'numSubjects', label: 'numSubjects', type: 'number', def: 20, help: 'Number of subjects analyzed.' }],

    boxes: [['entropy', 'Shannon entropy (bits)']] },

  { id: 'stft', group: 'Self-properties of UPEs', no: 5, title: 'Spectrogram (short FFT)', script: 'ShortFFT_UPEs.m', time: '5–10 min', kind: 'stft',

    aim: 'Spectrogram (time-frequency representation) of every electrode and subject trace using the short Fourier transform with a Hann window. The multitaper option is the paper’s remedy for the short FFT’s variance and leakage.',

    question: 'How does the frequency content of each signal change over time?',

    assumptions: ['The signal should not contain transient artifacts.'],

    pitfalls: ['Time-frequency trade-off.', 'Spectral leakage can persist even with windowing.'],

    notes: ['Longer windows improve frequency resolution and reduce time resolution. Try several windows and inspect (Figure 2).', 'More overlap (50–75%) gives smoother maps; use it for non-stationary signals.'],

    params: [

      { k: 'method', label: 'method', type: 'select', def: 'stft', opts: ['stft', 'multitaper'], help: 'stft = ShortFFT_UPEs.m; multitaper = Troubleshooting 1.' },

      { k: 'numSubjects', label: 'numSubjects', type: 'number', def: 20, help: 'Number of subjects analyzed.' },

      { k: 'fs', label: 'fs (Hz)', type: 'number', def: 20, help: 'Sampling frequency.' },

      { k: 'timeWindow', label: 'timeWindow (s)', type: 'number', def: 20, help: 'Length of each FFT segment.', when: { method: 'stft' } },

      { k: 'overlapLength', label: 'overlapLength', type: 'number', def: 0.75, help: 'Fraction of window overlap.', when: { method: 'stft' } },

      { k: 'minF', label: 'minF (Hz)', type: 'number', def: 0.001, help: 'Minimum frequency of interest.', when: { method: 'multitaper' } },

      { k: 'tw', label: 'time-half bandwidth', type: 'number', def: 3, help: 'TW, commonly 2–4.', when: { method: 'multitaper' } },

      { k: 'tapers', label: 'number of tapers', type: 'number', def: 5, help: 'Typically 2×TW − 1.', when: { method: 'multitaper' } },

      { k: 'win', label: 'window (s)', type: 'number', def: 10, help: 'Window size.', when: { method: 'multitaper' } },

      { k: 'step', label: 'step (s)', type: 'number', def: 2, help: 'Step size.', when: { method: 'multitaper' } },

      { k: 'detrend', label: 'detrend', type: 'select', def: 'off', opts: ['off', 'linear', 'constant'], help: 'Detrending of each segment.', when: { method: 'multitaper' } },

      { k: 'weighting', label: 'weighting', type: 'select', def: 'adapt', opts: ['adapt', 'unity', 'eigen'], help: 'How taper spectra are combined.', when: { method: 'multitaper' } },

    ] },

  { id: 'plot_spec', group: 'Self-properties of UPEs', no: 6, title: 'Plot spectrograms', script: 'Plot_spectrogram_UPEs.m', time: '1 min', kind: 'plot_spec', needs: 'stft',

    aim: 'Plot the spectrograms of the traces you select, from 2/timeWindow Hz up to fs/2, power in dB.',

    question: 'What does the time-frequency content of these selected traces look like?',

    params: [

      { k: 'fs', label: 'fs (Hz)', type: 'number', def: 20, help: 'Sampling frequency.' },

      { k: 'timeWindow', label: 'timeWindow (s)', type: 'number', def: 20, help: 'Window used in step 5.' },

    ] },

  { id: 'dfa', group: 'Self-properties of UPEs', no: 7, title: 'Detrended fluctuation analysis', script: 'ComputeDFA_UPE_singleWindow / ComputeDFA_UPE_Loop', time: '15–20 min', kind: 'dfa',

    aim: 'DFA scaling exponent as a proxy of long-range temporal correlations. Pipeline: zero-mean signal, Hann-window FIR band-pass, amplitude envelope, DFA. First choose FitInterval, CalcInterval and the filter on one trace using white noise as a check, then run the loop on all traces.',

    question: 'Does the signal have long-range temporal correlations (i.e., memory)?',

    assumptions: ['Data long enough for estimation across several time scales.', 'Approximately stationary signal.', 'Filtering must preserve the temporal correlation of interest.'],

    pitfalls: ['Aggressive or poorly designed filters can introduce correlations.', 'The exponent can arise from different mechanisms, which limits mechanistic conclusions.'],

    notes: ['Below 0.5: anti-correlated. 0.5: uncorrelated (white-noise-like). 0.5–1: persistent long-range correlation. Above 1: strongly correlated or non-stationary.',

      'White-noise traces (n = 1000) should give 0.5 ± 0.03 in the FitInterval. FitInterval must lie inside CalcInterval.'],

    params: [

      { k: 'electrode', label: 'Electrode', type: 'field', def: '', help: 'Field to use for the single-window check.' },

      { k: 'subject', label: 'trace (subject)', type: 'number', def: 1, help: 'Subject column to use.' },

      { k: 'hp', label: 'hp (Hz)', type: 'number', def: 0.1, help: 'High-pass edge.' },

      { k: 'lp', label: 'lp (Hz)', type: 'number', def: 1, help: 'Low-pass edge.' },

      { k: 'fs', label: 'fs (Hz)', type: 'number', def: 20, help: 'Sampling frequency.' },

      { k: 'orderFactor', label: 'filter_order = X / hp', type: 'number', def: 50, help: 'Paper uses 50/hp.' },

      { k: 'fit', label: 'FitInterval (s)', type: 'pair', def: [80, 150], help: 'Time scales used for the power-law fit.' },

      { k: 'calc', label: 'CalcInterval (s)', type: 'pair', def: [20, 200], help: 'Time-window sizes computed.' },

      { k: 'overlap', label: 'overlap', type: 'number', def: 0.8, help: 'Overlap between consecutive windows.' },

      { k: 'nWhite', label: 'white-noise traces', type: 'number', def: 100, help: 'Paper uses 1000 (slow).' },

      { k: 'numSubjects', label: 'numSubjects', type: 'number', def: 20, help: 'Used by the loop.' },

    ] },

  { id: 'correlation', group: 'Functional interactions', no: 8, title: 'Correlation', script: 'ComputeCorrelationUPEs', time: '5–10 min', kind: 'net',

    aim: 'Pairwise correlation between electrodes in sliding windows, as a proxy of temporal synchronicity.',

    question: 'How synchronized in time are the traces of each electrode pair?',

    assumptions: ['Signal free from artifacts.', 'Stationary signal.', 'Approximately normal values if Pearson’s correlation is used.'],

    pitfalls: ['Cannot separate direct from indirect connections.', 'Sensitive to volume conduction, third-party effects and outliers.', 'Assumes linear interaction.'],

    notes: [NET_NOTE, 'Partial correlation reduces common-source effects.'],

    params: [...NET_PARAMS, { k: 'partial', label: 'partial correlation', type: 'bool', def: false, help: 'Control for the other signals (partialcorr).' }],

    unit: 'Correlation coefficient' },

  { id: 'coherence', group: 'Functional interactions', no: 9, title: 'Coherence', script: 'ComputeCoherenceUPEs', time: '5–10 min', kind: 'net',

    aim: 'Median pairwise coherence across all frequencies between electrodes, as a proxy of frequency synchronicity.',

    question: 'How consistently do two signals oscillate together at specific frequencies?',

    assumptions: ['Weakly non-stationary within each window.', 'fs high enough for the frequencies of interest (Nyquist).', 'TimeWindow long enough for robust spectral estimation.'],

    pitfalls: ['Sensitive to volume conduction and common reference.', 'Cannot separate direct from indirect connections.'],

    notes: [NET_NOTE, 'Partial directed coherence gives direction and removes shared sources.'],

    params: NET_PARAMS, unit: 'Coherence' },

  { id: 'granger', group: 'Functional interactions', no: 10, title: 'Granger causality', script: 'ComputeGrangerCausalityUPEs', time: '15–20 min', kind: 'net',

    aim: 'Directed pairwise Granger causality between electrodes; the remaining trace is the conditioning variable. Edges with p > 0.05 are set to zero. Order matters: BO is Background to Occipital.',

    question: 'Does one signal help predict the future of another?',

    assumptions: ['Stationary signal.', 'Signal long enough relative to the model order.'],

    pitfalls: ['Susceptible to false positives.', 'Predictive power is not mechanistic causation.', 'Sensitive to model order (1 lag here).'],

    notes: [NET_NOTE, 'Optional: phase-randomized surrogates (> 250) for an empirical p-value, then time-reversed GC. TRGC index = (GCfwd − GCrev) / (GCfwd + GCrev); above 0.3 suggests asymmetric, more robust causality.'],

    params: [...NET_PARAMS,

      { k: 'nSurrogates', label: 'surrogates', type: 'number', def: 0, help: '0 = off. Use > 250 for reliable significance.' },

      { k: 'trgc', label: 'time-reversed GC', type: 'bool', def: false, help: 'Compute the TRGC index for edges that pass.' },
      { k: 'corrected', label: 'Use paper-described GC windows/cutoff', type: 'bool', def: false, help: 'Off preserves the uploaded MATLAB-script behaviour; on uses fixed-length windows and the stated p < 0.05 rule.' }],

    unit: 'Granger test statistic', log: true },

  { id: 'mutual_info', group: 'Functional interactions', no: 11, title: 'Mutual information', script: 'ComputeMutualInformationUPEs', time: '5–10 min', kind: 'net',

    aim: 'Pairwise mutual information between electrodes as a proxy of shared information, capturing linear and nonlinear dependency.',

    question: 'How much does knowing Signal A reduce the uncertainty about Signal B, linear or not?',

    assumptions: ['Stationary signal.'],

    pitfalls: ['Sensitive to the choice of k nearest neighbors.', 'Does not give directionality.'],

    notes: [NET_NOTE],

    params: [...NET_PARAMS, { k: 'K', label: 'K', type: 'number', def: 3, help: 'k-th nearest neighbor.' }], unit: 'Mutual information' },

  { id: 'stats', group: 'Statistics', no: 12, title: 'Mixed-effects model', script: 'MixedEffectModels.R', time: 'variable', kind: 'stats',

    aim: 'Linear mixed-effects model for a metric by electrode and task (20 subjects × 3 electrodes × 5 tasks, repeated measures), with a Type III ANOVA and post-hoc pairwise contrasts with Cohen’s d.',

    question: 'Does the metric differ by electrode, by task, or by their interaction, after accounting for each subject?',

    pitfalls: ['More random effects raise the risk of singular fits.', 'Singular models usually mean overfitting; simplify the random structure.'],

    notes: ['Dataset columns: Subject, Electrode, Task and the metric (for example Entropy).', 'Compare candidate formulas by AIC, BIC and logLik; singular fits are dropped.'],

    params: [

      { k: 'dv', label: 'dependent variable', type: 'text', def: 'Entropy', help: 'Column to model.' },

      { k: 'formula', label: 'formula', type: 'text', def: 'Entropy ~ Electrode * Task + (1 | Subject) + (1 | Subject:Task)', help: 'lmer formula.' },

      { k: 'correction', label: 'post-hoc correction', type: 'select', def: 'bonferroni', opts: ['bonferroni', 'tukey', 'holm', 'fdr'], help: 'Multiple-comparison adjustment.' },

      { k: 'contrast', label: 'post-hoc comparison', type: 'select', def: 'Electrode|Task', opts: ['Electrode', 'Electrode|Task', 'Task|Electrode'], help: 'Which pairwise comparisons.' },

    ] },

];

const GROUPS = ['Start', 'Self-properties of UPEs', 'Functional interactions', 'Statistics'];



/* ----------------------------- small UI helpers ----------------------------- */

const LAYOUT = { autosize: true, font: { family: 'inherit', size: 12, color: '#1c2430' }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: { l: 55, r: 15, t: 25, b: 45 }, height: 320 };

const Fig = ({ data, layout, h }) => (

  <Plot data={data} layout={{ ...LAYOUT, ...(h ? { height: h } : {}), ...layout }} config={{ displaylist: false, responsive: true }} style={{ width: '100%' }} useResizeHandler />

);

const Card = ({ title, children, right }) => (

  <section className="border border-slate-200 bg-white rounded-xl">

    {title && <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5"><h3 className="text-sm font-semibold text-slate-900">{title}</h3>{right}</div>}

    <div className="p-4">{children}</div>

  </section>

);

const Bullets = ({ items }) => <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">{items.map((t, i) => <li key={i}>{t}</li>)}</ul>;

const Table = ({ rows }) => {

  if (!rows?.length) return null;

  const cols = Object.keys(rows[0]);

  return (

    <div className="overflow-x-auto"><table className="w-full text-xs">

      <thead><tr>{cols.map((c) => <th key={c} className="border-b border-slate-300 px-2 py-1.5 text-left font-semibold text-slate-700">{c}</th>)}</tr></thead>

      <tbody>{rows.map((r, i) => <tr key={i} className="odd:bg-slate-50">{cols.map((c) => <td key={c} className="px-2 py-1.5 font-mono text-slate-700">{typeof r[c] === 'number' ? +r[c].toPrecision(4) : String(r[c])}</td>)}</tr>)}</tbody>

    </table></div>

  );

};

const Downloads = ({ items }) => items?.length ? (

  <div className="flex flex-wrap gap-2">{items.map((d) => (

    <a key={d.url} href={API_URL + d.url} className="inline-flex items-center gap-1.5 border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"><Download className="h-3.5 w-3.5" />{d.label}</a>

  ))}</div>

) : null;

const Boxes = ({ data, ylabel, log }) => (

  <Fig data={Object.entries(data).map(([name, y]) => ({ type: 'box', name, y, boxpoints: 'all', jitter: 0.4, pointpos: 0, marker: { size: 5 }, line: { width: 1.5 } }))}

    layout={{ showlegend: false, yaxis: { title: ylabel, type: log ? 'log' : 'linear' } }} />

);

const Heat = ({ z, x, y, scale = 'Viridis', title, h = 320, rev }) => (

  <Fig h={h} data={[{ type: 'heatmap', z, x, y, colorscale: scale, colorbar: { title: { text: title }, thickness: 12 } }]} layout={{ yaxis: { autorange: rev ? 'reversed' : true } }} />

);



function ParamForm({ defs, values, set, fields }) {

  const visible = defs.filter((d) => !d.when || Object.entries(d.when).every(([k, v]) => values[k] === v));

  const cls = 'w-full border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 focus:border-slate-900 focus:outline-none';

  return (

    <div className="grid grid-cols-1 gap-x-5 gap-y-3 sm:grid-cols-2 xl:grid-cols-3">

      {visible.map((d) => (

        <label key={d.k} className="flex flex-col gap-1">

          <span className="font-mono text-xs font-semibold text-slate-800">{d.label}</span>

          {d.type === 'select' && <select className={cls} value={values[d.k]} onChange={(e) => set(d.k, e.target.value)}>{d.opts.map((o) => <option key={o}>{o}</option>)}</select>}

          {d.type === 'field' && <select className={cls} value={values[d.k]} onChange={(e) => set(d.k, e.target.value)}>{(fields || []).map((o) => <option key={o}>{o}</option>)}</select>}

          {d.type === 'bool' && <input type="checkbox" className="h-4 w-4 self-start" checked={!!values[d.k]} onChange={(e) => set(d.k, e.target.checked)} />}

          {(d.type === 'number' || d.type === 'text') && <input className={cls} type={d.type === 'number' ? 'number' : 'text'} step="any" value={values[d.k]} onChange={(e) => set(d.k, e.target.value)} />}

          {d.type === 'pair' && <div className="flex gap-2">{[0, 1].map((i) => <input key={i} className={cls} type="number" step="any" value={values[d.k][i]} onChange={(e) => { const v = [...values[d.k]]; v[i] = e.target.value; set(d.k, v); }} />)}</div>}

          <span className="text-xs text-slate-500">{d.help}</span>

        </label>

      ))}

    </div>

  );

}



/* ----------------------------- result renderers ----------------------------- */

function Clustering({ r }) {

  const n = r.z.length, ax = { range: [n - 0.5, -0.5], showticklabels: false, zeroline: false };

  const segs = r.dendrogram.icoord.map((ic, i) => ({ type: 'scatter', mode: 'lines', x: r.dendrogram.dcoord[i].map((d) => -d), y: ic.map((v) => (v - 5) / 10), line: { color: '#475569', width: 1 }, hoverinfo: 'skip' }));

  return (

    <div className="space-y-4">

      <div className="flex">

        <div className="w-1/5"><Fig h={520} data={segs} layout={{ showlegend: false, margin: { l: 5, r: 0, t: 25, b: 45 }, yaxis: ax, xaxis: { showticklabels: false, zeroline: false } }} /></div>

        <div className="w-4/5"><Fig h={520} data={[{ type: 'heatmap', z: r.z, y: r.row_labels.map((_, i) => i), text: r.row_labels.map((l) => Array(r.z[0].length).fill(l)), hovertemplate: '%{text}<br>t=%{x}<br>%{z:.2f}<extra></extra>', colorscale: 'RdBu', reversescale: true, colorbar: { title: { text: 'Norm. value' }, thickness: 12 } }]}

          layout={{ margin: { l: 5, r: 15, t: 25, b: 45 }, yaxis: ax, xaxis: { title: 'time point' } }} /></div>

      </div>

      <div className="grid gap-4 md:grid-cols-2">

        <div><p className="mb-1 text-xs font-medium text-slate-600">Cluster of each row (in plotted order)</p>

          <Fig h={220} data={[{ type: 'scatter', mode: 'markers', x: r.clusters, y: r.row_labels, marker: { color: r.clusters, colorscale: 'Portland', size: 7 } }]} layout={{ xaxis: { title: 'cluster', dtick: 1 }, yaxis: { autorange: 'reversed', showticklabels: false } }} /></div>

        {r.silhouette && <div><p className="mb-1 text-xs font-medium text-slate-600">Silhouette score vs number of clusters</p>

          <Fig h={220} data={[{ type: 'scatter', mode: 'lines+markers', x: r.silhouette.k, y: r.silhouette.score, line: { color: ACCENT } }]} layout={{ xaxis: { title: 'k', dtick: 1 }, yaxis: { title: 'silhouette' } }} /></div>}

      </div>

    </div>

  );

}



function Stft({ r }) {

  return (

    <div className="space-y-4">

      {r.trace_names && <p className="text-sm text-slate-600">{r.trace_names.length} spectrograms computed (power matrix {r.shape?.join(' × ')}). Use step 6 to view them.</p>}

      {r.avg_power && <Fig h={260} data={[{ type: 'scatter', mode: 'lines', x: r.avg_power.f, y: r.avg_power.p, line: { color: ACCENT } }]} layout={{ xaxis: { title: 'Frequency (Hz)', type: 'log' }, yaxis: { title: 'Avg. power (dB)' }, title: { text: r.avg_power.name, font: { size: 12 } } }} />}

      {r.compare?.map((c) => (

        <div key={c.label} className="grid gap-3 md:grid-cols-2">

          <Fig h={240} data={[{ type: 'heatmap', x: c.t, y: c.f, z: c.z, colorscale: 'Jet', colorbar: { title: { text: 'dB' }, thickness: 10 } }]} layout={{ title: { text: c.label, font: { size: 12 } }, xaxis: { title: 'Time (s)' }, yaxis: { title: 'Hz' } }} />

          <Fig h={240} data={[{ type: 'scatter', mode: 'lines', x: c.avg_f, y: c.avg_p, line: { color: ACCENT } }]} layout={{ xaxis: { title: 'Frequency (Hz)', type: 'log' }, yaxis: { title: 'Avg. power (dB)' } }} />

        </div>

      ))}

    </div>

  );

}



function Dfa({ r }) {

  const t = r.trace, w = r.white;

  return (

    <div className="space-y-4">

      {r.filter && <Card title="Filter frequency response (check: flat passband, smooth roll-off, no ripples)"><Fig h={240} data={[{ type: 'scatter', mode: 'lines', x: r.filter.f, y: r.filter.mag_db, line: { color: ACCENT } }]} layout={{ xaxis: { title: 'Frequency (Hz)' }, yaxis: { title: 'Magnitude (dB)' } }} /></Card>}

      {w && <Card title="White-noise check">

        <p className="mb-2 text-sm text-slate-700">Mean DFA exponent of {w.n} white-noise traces: <b className="font-mono">{w.mean.toFixed(3)}</b> (sd {w.sd.toFixed(3)}) — {w.ok ? 'within 0.5 ± 0.03, the pipeline is valid for this FitInterval.' : 'outside 0.5 ± 0.03. Change the FitInterval or the filter before using the loop.'}</p>

        <Fig h={240} data={[{ type: 'scatter', mode: 'lines+markers', x: w.slopes.x, y: w.slopes.y, line: { color: ACCENT } }]}

          layout={{ xaxis: { title: 'log10 window (s)' }, yaxis: { title: 'local exponent' }, shapes: [{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 0.47, y1: 0.53, fillcolor: '#16a34a', opacity: 0.15, line: { width: 0 } }] }} />

      </Card>}

      {t && <Card title={`Trace DFA exponent α = ${t.exponent.toFixed(3)}`}>

        <div className="grid gap-3 md:grid-cols-2">

          <Fig h={260} data={[{ type: 'scatter', mode: 'lines', y: t.signal, name: 'filtered', line: { color: '#94a3b8', width: 1 } }, { type: 'scatter', mode: 'lines', y: t.envelope, name: 'envelope', line: { color: ACCENT, width: 1.5 } }]} layout={{ xaxis: { title: 'sample' }, legend: { orientation: 'h' } }} />

          <Fig h={260} data={[{ type: 'scatter', mode: 'markers', x: t.x_all, y: t.y_all, marker: { color: '#475569', size: 6 } }, { type: 'scatter', mode: 'markers', x: t.x_fit, y: t.y_fit, marker: { color: '#dc2626', size: 8 } }, { type: 'scatter', mode: 'lines', x: t.x_fit, y: t.line_y, line: { color: '#dc2626' } }]} layout={{ showlegend: false, xaxis: { title: 'log10 window (s)' }, yaxis: { title: 'log10 <F(t)>' } }} />

        </div>

      </Card>}

      {r.loop && <Card title="DFA scaling exponent, all traces"><Boxes data={Object.fromEntries(r.loop.cols.map((c, j) => [c, r.loop.values.map((row) => row[j])]))} ylabel="DFA scaling exponent" /></Card>}

    </div>

  );

}



function Net({ r, def }) {

  const [subj, setSubj] = useState(0);

  const nSub = Object.values(r.trajectories || {})[0]?.length || 0;

  return (

    <div className="space-y-4">

      <Card title="Median across windows, one point per subject"><Boxes data={r.median} ylabel={def.unit} log={def.log} /></Card>

      {r.trajectories && <Card title="Time-resolved edge values" right={<select className="border border-slate-300 px-2 py-1 text-xs" value={subj} onChange={(e) => setSubj(+e.target.value)}>{Array.from({ length: nSub }, (_, i) => <option key={i} value={i}>subject {i + 1}</option>)}</select>}>

        <Fig h={260} data={Object.entries(r.trajectories).map(([p, v]) => ({ type: 'scatter', mode: 'lines', name: p, y: v[subj] }))} layout={{ xaxis: { title: 'window' }, yaxis: { title: def.unit } }} /></Card>}

      {r.mean_network && <Card title="Mean network over windows and subjects"><Heat z={r.mean_network.z} x={r.mean_network.labels} y={r.mean_network.labels} title={def.unit} h={300} rev /></Card>}

      {r.trgc && <Card title="Time-reversed Granger causality"><Table rows={Object.entries(r.trgc).map(([pair, v]) => ({ pair, mean_index: v.index.reduce((a, b) => a + b, 0) / (v.index.length || 1), fraction_above_0_3: v.fraction_above }))} /></Card>}

    </div>

  );

}



function Stats({ r }) {

  return (

    <div className="space-y-4">

      {r.singular && <div className="flex gap-2 border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900"><AlertCircle className="h-4 w-4 flex-shrink-0" />Singular fit: the random-effects structure may be too complex. Simplify the formula.</div>}

      {r.summary && <Card title="Model summary"><pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs text-slate-800">{r.summary}</pre></Card>}

      {r.anova && <Card title="Type III Wald tests (chi-square; Python engine)"><Table rows={r.anova} /></Card>}

      {r.posthoc && <Card title="Post-hoc pairwise comparisons"><Table rows={r.posthoc} /></Card>}

      {r.compare && <Card title="Model comparison (sorted by AIC)"><Table rows={r.compare} /></Card>}

    </div>

  );

}


function DatasetPreview({ res }) {
  const fields = res.fields || [];
  const [field, setField] = useState(fields[0] || '');
  const [subj, setSubj] = useState(0);
  const [overlay, setOverlay] = useState(true);

  // reset subject selector when switching field
  useEffect(() => { setSubj(0); }, [field]);

  const raw = res.preview?.[field];
  // tolerate both number[][] (new) and number[] (legacy) shapes
  const multi = Array.isArray(raw?.[0]);
  const subjects = multi ? raw : raw ? [raw] : [];
  const nSub = subjects.length;

  const traces = overlay
    ? subjects.map((y, i) => ({
        type: 'scatter', mode: 'lines', y,
        name: `subject ${i + 1}`,
        line: { width: 0.7, color: 'rgba(51,71,201,0.35)' },
        showlegend: false,
        hovertemplate: `subject ${i + 1}<br>t=%{x}<br>y=%{y:.3f}<extra></extra>`,
      }))
    : [{
        type: 'scatter', mode: 'lines',
        y: subjects[subj] || [],
        name: `subject ${subj + 1}`,
        line: { color: ACCENT, width: 1.3 },
      }];

  return (
    <Card
      title={`Traces — ${field} (${nSub} subject${nSub === 1 ? '' : 's'})`}
      right={
        <div className="flex items-center gap-3 text-xs">
          <select
            className="border border-slate-300 px-2 py-1 text-xs"
            value={field}
            onChange={(e) => setField(e.target.value)}
          >
            {fields.map((f) => <option key={f}>{f}</option>)}
          </select>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              className="h-3.5 w-3.5"
              checked={overlay}
              onChange={(e) => setOverlay(e.target.checked)}
            />
            overlay all
          </label>
          {!overlay && (
            <select
              className="border border-slate-300 px-2 py-1 text-xs"
              value={subj}
              onChange={(e) => setSubj(+e.target.value)}
            >
              {subjects.map((_, i) => (
                <option key={i} value={i}>subject {i + 1}</option>
              ))}
            </select>
          )}
        </div>
      }
    >
      <Fig
        h={320}
        data={traces}
        layout={{ xaxis: { title: 'time point' }, yaxis: { title: 'UPE count' }, showlegend: false }}
      />
    </Card>
  );
}


/* ----------------------------------- app ----------------------------------- */

export default function App() {

  const [active, setActive] = useState('about');

  const [sid, setSid] = useState(null);

  const [info, setInfo] = useState(null);
  const [selectedTask, setSelectedTask] = useState('');

  const [files, setFiles] = useState([]);

  const [params, setParams] = useState(() => Object.fromEntries(STEPS.filter((s) => s.params).map((s) => [s.id, Object.fromEntries(s.params.map((p) => [p.k, p.def]))])));

  const [results, setResults] = useState({});

  const [busy, setBusy] = useState(null);

  const [errors, setErrors] = useState({});

  const [compareWin, setCompareWin] = useState('10, 30, 60');

  const [picked, setPicked] = useState([]);

  const [statsFile, setStatsFile] = useState(null);

  const fileRef = useRef(null);

  const statsRef = useRef(null);



  const step = STEPS.find((s) => s.id === active);

  const setP = (id) => (k, v) => setParams((p) => ({ ...p, [id]: { ...p[id], [k]: v } }));

  const fail = (id, e) => setErrors((x) => ({ ...x, [id]: e.response?.data?.detail || e.message }));



  const clean = (id) => Object.fromEntries(Object.entries(params[id] || {}).map(([k, v]) => {

    const d = STEPS.find((s) => s.id === id).params.find((p) => p.k === k);

    if (d?.type === 'number') return [k, Number(v)];

    if (d?.type === 'pair') return [k, v.map(Number)];

    return [k, v];

  }));



  const upload = async () => {

    const fd = new FormData();

    files.forEach((f) => fd.append('files', f));

    setBusy('dataset'); setErrors({});

    try {

      const { data } = await api.post('/sessions', fd);

      setSid(data.session_id); setInfo(data); setSelectedTask(data.tasks?.[0] || ''); setResults({ dataset: data });

      setParams((p) => {

        const next = { ...p };

        Object.keys(next).forEach((id) => { if ('numSubjects' in next[id]) next[id] = { ...next[id], numSubjects: data.n_subjects }; });

        if (next.dfa) next.dfa = { ...next.dfa, electrode: data.fields[0] };

        return next;

      });

    } catch (e) { fail('dataset', e); } finally { setBusy(null); }

  };



  const uploadStats = async (file) => {

    setStatsFile(file);

    const fd = new FormData(); fd.append('file', file);

    try { const { data } = await api.post(`/sessions/${sid}/stats_dataset`, fd); setResults((r) => ({ ...r, stats_upload: data })); }

    catch (e) { fail('stats', e); }

  };



  const run = async (id, action = 'run', extra = {}) => {

    setBusy(id + action); setErrors((x) => ({ ...x, [id]: null }));

    try {

      const { data } = await api.post(`/sessions/${sid}/steps/${id}`, { action, params: { ...clean(id), ...(selectedTask ? { task: selectedTask } : {}), ...extra } });

      setResults((r) => ({ ...r, [id]: { ...(r[id] || {}), ...data } }));

      if (id === 'stft' && data.trace_names) setPicked(data.trace_names.slice(0, 1));

    } catch (e) { fail(id, e); } finally { setBusy(null); }

  };



  const res = results[active];

  const locked = (s) => s.id !== 'about' && s.id !== 'dataset' && !sid;

  const needsMissing = step.needs && !results[step.needs];

  const Btn = ({ label, onClick, action, primary }) => (

    <button onClick={onClick} disabled={!!busy || locked(step) || needsMissing}

      className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40 ${primary ? 'text-white' : 'border border-slate-300 text-slate-800 hover:bg-slate-50'}`}

      style={primary ? { background: ACCENT } : {}}>

      {busy === step.id + action ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-3.5 w-3.5" />}{label}

    </button>

  );



  return (

    <div className="flex h-screen w-screen overflow-hidden bg-slate-100 text-slate-900 antialiased">

      <aside className="flex w-72 flex-shrink-0 flex-col overflow-y-auto border-r border-slate-200 bg-white">

        <div className="border-b border-slate-200 px-5 py-4">

          <h1 className="text-base font-semibold leading-tight">Photoencephalography analysis</h1>

          <p className="mt-1 text-xs text-slate-500">Python version of the STAR Protocols data analysis (Bonzanni, Rouleau, Murugan 2025)</p>

        </div>

        <nav className="flex-1 py-2">

          {GROUPS.map((g) => (

            <div key={g} className="mb-2">

              <p className="px-5 py-1.5 text-xs font-medium text-slate-500">{g}</p>

              {STEPS.filter((s) => s.group === g).map((s) => {

                const done = s.id !== 'about' && results[s.id], on = s.id === active;

                return (

                  <button key={s.id} onClick={() => setActive(s.id)} className={`flex w-full items-center gap-3 px-5 py-2 text-left text-sm ${on ? 'bg-slate-100 font-semibold' : 'hover:bg-slate-50'} ${locked(s) ? 'text-slate-400' : 'text-slate-800'}`}

                    style={on ? { boxShadow: `inset 3px 0 0 ${ACCENT}` } : {}}>

                    <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full border text-[11px]" style={done ? { background: ACCENT, borderColor: ACCENT, color: '#fff' } : { borderColor: '#cbd5e1' }}>

                      {done ? <Check className="h-3 w-3" /> : locked(s) ? <Lock className="h-2.5 w-2.5" /> : s.no ?? '·'}

                    </span>

                    <span className="truncate">{s.title}</span>

                  </button>

                );

              })}

            </div>

          ))}

        </nav>
        <div className="border-t border-slate-200 px-5 py-3 text-xs text-slate-500">
          {info ? <>
            <p>{info.fields.length} fields · {info.n_subjects} subjects · {info.n_points} points</p>
            {info.tasks?.length > 0 && <label className="mt-3 flex flex-col gap-1">
              <span className="font-medium text-slate-600">Task (analyses run separately)</span>
              <select value={selectedTask} onChange={(e) => {
                const value = e.target.value;
                setSelectedTask(value);
                const matching = info.fields.find((name) => name.endsWith(`_${value}`));
                const electrode = matching ? matching.slice(0, -(value.length + 1)) : '';
                setParams((p) => ({ ...p, dfa: { ...p.dfa, electrode } }));
                setResults((r) => ({ dataset: r.dataset, stats_upload: r.stats_upload }));
                setPicked([]);
              }} className="w-full border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-800">
                {info.tasks.map((task) => <option key={task} value={task}>{task}</option>)}
              </select>
            </label>}
          </> : 'No dataset loaded'}
        </div>

      </aside>



      <main className="flex-1 overflow-y-auto">

        <div className="mx-auto max-w-5xl space-y-5 p-6">

          {step.kind === 'about' ? (

            <Card title="How this app follows the paper">

              <div className="space-y-3 text-sm text-slate-700">

                <p>Each step is one script of the paper’s Data analysis section, re-implemented in Python. Load your dataset in step 1, then run any step. Every step shows the parameters the MATLAB script uses, the outputs it produces, and the question those outputs answer.</p>

                <p>Steps 2–7 describe the self-properties of the UPE traces. Steps 8–11 describe interactions between electrodes. Step 12 tests the metrics statistically.</p>

                <p>Expected outcome (paper Figure 3): entropy and CV higher for brain electrodes than background, more very-low-frequency power (&lt; 1 Hz), higher DFA exponents, more correlation and mutual information within brain regions, no clear coherence, and occipital driving temporal in Granger causality.</p>

                <button onClick={() => setActive('dataset')} className="px-4 py-2 text-sm font-medium text-white" style={{ background: ACCENT }}>Start with the dataset</button>

              </div>

            </Card>

          ) : (

            <>

              <header>

                <h2 className="text-xl font-semibold">Step {step.no}: {step.title}</h2>

                <p className="mt-1 font-mono text-xs text-slate-500">{step.script} · about {step.time}</p>

                <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-700">{step.aim}</p>

              </header>



              {step.question && <div className="border-l-4 bg-white px-4 py-3" style={{ borderColor: ACCENT }}><p className="text-xs font-medium text-slate-500">The question this step answers</p><p className="mt-0.5 text-sm font-medium text-slate-900">{step.question}</p></div>}



              {step.kind === 'dataset' ? (

                <Card title="Load data">

                  <input ref={fileRef} type="file" multiple accept=".xlsx,.mat,.csv" className="hidden" onChange={(e) => {
  const chosen = Array.from(e.target.files || []);
  e.target.value = ''; // safe now — we already copied the File objects
  if (!chosen.length) return;
  setFiles((prev) => [
    ...prev,
    ...chosen.filter((n) => !prev.some((x) => x.name === n.name)),
  ]);
}} />

                  <button onClick={() => fileRef.current.click()} className="flex w-full flex-col items-center gap-2 border border-dashed border-slate-300 py-6 text-sm text-slate-600 hover:bg-slate-50"><Upload className="h-5 w-5" />Choose an Excel, CSV, or UPEs.mat file</button>

                  {files.length > 0 && <ul className="mt-3 space-y-1 text-sm">{files.map((f) => <li key={f.name} className="flex justify-between border border-slate-200 px-2.5 py-1.5"><span>{f.name}</span><button className="text-xs text-slate-500 hover:text-slate-900" onClick={() => setFiles(files.filter((x) => x !== f))}>remove</button></li>)}</ul>}

                  <div className="mt-4"><Btn primary label="Load dataset" action="" onClick={upload} /></div>

                </Card>

              ) : (

                <Card title="Parameters">

                  <ParamForm defs={step.params} values={params[step.id]} set={setP(step.id)} fields={step.kind === 'dfa' && selectedTask && info?.fields ? info.fields.filter((name) => name.endsWith(`_${selectedTask}`)).map((name) => name.slice(0, -(selectedTask.length + 1))) : info?.fields} />

                  {step.kind === 'plot_spec' && (

                    <label className="mt-4 flex flex-col gap-1"><span className="font-mono text-xs font-semibold">traces (listdlg)</span>

                      <select multiple size={6} className="border border-slate-300 px-2 py-1 text-sm" value={picked} onChange={(e) => setPicked(Array.from(e.target.selectedOptions, (o) => o.value))}>{(results.stft?.trace_names || []).map((n) => <option key={n}>{n}</option>)}</select></label>

                  )}

                  {step.kind === 'stats' && (

                    <div className="mt-4"><input ref={statsRef} type="file" accept=".xlsx" className="hidden" onChange={(e) => e.target.files[0] && uploadStats(e.target.files[0])} />

                      <button disabled={!sid} onClick={() => statsRef.current.click()} className="inline-flex items-center gap-2 border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50 disabled:opacity-40"><Upload className="h-4 w-4" />{statsFile ? statsFile.name : 'Upload stats dataset (.xlsx)'}</button>

                      {results.stats_upload && <span className="ml-3 text-xs text-slate-500">{results.stats_upload.n_rows} rows · {results.stats_upload.columns.join(', ')}</span>}</div>

                  )}

                  <div className="mt-5 flex flex-wrap items-center gap-3">

                    {step.kind === 'dfa' ? (<>

                      <Btn action="filter" label="Preview filter" onClick={() => run('dfa', 'filter')} />

                      <Btn primary action="single" label="Check one trace + white noise" onClick={() => run('dfa', 'single')} />

                      <Btn action="loop" label="Run all traces" onClick={() => run('dfa', 'loop')} />

                    </>) : step.kind === 'plot_spec' ? <Btn primary action="run" label="Plot selected" onClick={() => run('plot_spec', 'run', { traces: picked })} />

                      : step.kind === 'stats' ? (<>

                        <Btn primary action="fit" label="Fit model" onClick={() => run('stats', 'fit')} />

                        <Btn action="compare" label="Compare candidate models" onClick={() => run('stats', 'compare')} />

                      </>) : <Btn primary action="run" label="Run step" onClick={() => run(step.id)} />}

                    {step.kind === 'stft' && <span className="flex items-center gap-2 text-sm">

                      <input className="w-32 border border-slate-300 px-2 py-1.5 text-sm" value={compareWin} onChange={(e) => setCompareWin(e.target.value)} />

                      <Btn action="compare" label="Compare windows (s)" onClick={() => run('stft', 'compare', { windows: compareWin.split(',').map((x) => Number(x)).filter(Boolean) })} /></span>}

                    {needsMissing && <span className="text-xs text-slate-500">Run step {STEPS.find((s) => s.id === step.needs).no} first.</span>}

                    {!sid && <span className="text-xs text-slate-500">Load a dataset in step 1 first.</span>}

                  </div>

                </Card>

              )}



              {errors[step.id] && <div className="flex gap-2 border border-rose-300 bg-rose-50 p-3 text-sm text-rose-900"><AlertCircle className="h-4 w-4 flex-shrink-0" />{String(errors[step.id])}</div>}

              {res?.warnings?.map((w) => <div key={w} className="border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">{w}</div>)}



              {res && (

                <div className="space-y-4">

                  <h3 className="text-sm font-semibold">Output</h3>

{step.kind === 'dataset' && res.fields && (
  <>
    <Card title="Dataset summary">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <div className="text-xs text-slate-500">Subjects</div>
          <div className="text-2xl font-semibold" style={{ color: ACCENT }}>{res.n_subjects}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500">Fields</div>
          <div className="text-2xl font-semibold">{res.fields.length}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500">Time points</div>
          <div className="text-2xl font-semibold">{res.n_points}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500">Tasks</div>
          <div className="text-2xl font-semibold">{res.tasks?.length ?? 0}</div>
        </div>
      </div>
      <p className="mt-3 text-xs text-slate-500">
        Fields: <span className="font-mono">{res.fields.join(', ')}</span>
      </p>
    </Card>
    <DatasetPreview res={res} />
  </>
)}
                  {step.kind === 'clustering' && res.z && <Clustering r={res} />}

                  {step.kind === 'box' && step.boxes.map(([key, label]) => res[key] && <Card key={key} title={label}><Boxes data={res[key]} ylabel={label} /></Card>)}

                  {step.id === 'entropy' && res.bin_width !== undefined && <p className="text-sm text-slate-600">Unified Freedman-Diaconis bin width: <b className="font-mono">{res.bin_width}</b></p>}

                  {step.kind === 'stft' && <Stft r={res} />}

                  {step.kind === 'plot_spec' && res.plots?.map((p) => <Card key={p.name} title={p.name}><Fig h={300} data={[{ type: 'heatmap', x: p.t, y: p.f, z: p.z, colorscale: 'Jet', colorbar: { title: { text: 'Power (dB)' }, thickness: 12 } }]} layout={{ xaxis: { title: 'Time (s)' }, yaxis: { title: 'Frequency (Hz)' } }} /></Card>)}

                  {step.kind === 'dfa' && <Dfa r={res} />}

                  {step.kind === 'net' && res.median && <Net r={res} def={step} />}

                  {step.kind === 'stats' && <Stats r={res} />}

                  <Downloads items={res.downloads} />

                </div>

              )}



              {(step.assumptions || step.pitfalls || step.notes) && (

                <Card title="Reading the result">

                  <div className="grid gap-5 md:grid-cols-2">

                    {step.assumptions && <div><p className="mb-1.5 text-sm font-semibold">Assumptions</p><Bullets items={step.assumptions} /></div>}

                    {step.pitfalls && <div><p className="mb-1.5 text-sm font-semibold">Pitfalls</p><Bullets items={step.pitfalls} /></div>}

                  </div>

                  {step.notes && <div><p className="mb-1.5 text-sm font-semibold">Notes</p><Bullets items={step.notes} /></div>}

                </Card>

              )}

            </>

          )}

        </div>

      </main>

    </div>

  );

}