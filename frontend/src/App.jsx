import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Play, Settings, Activity, Network, BarChart3, ArrowRight, Database, BrainCircuit, LineChart as ChartIcon } from 'lucide-react'

// SVG Graph Component for Adjacency Matrix (Modern Theme)
const CausalGraph = ({ adj }) => {
  if (!adj || adj.length === 0) return <div className="placeholder" style={{color: 'white', textAlign:'center', marginTop:'40px'}}>No Graph Data</div>;
  const size = 300;
  const center = size / 2;
  const radius = 100;
  const n = adj.length;

  const nodes = Array.from({ length: n }).map((_, i) => {
    const angle = (i * 2 * Math.PI) / n - Math.PI / 2;
    return {
      id: i,
      x: center + radius * Math.cos(angle),
      y: center + radius * Math.sin(angle),
    };
  });

  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (adj[i][j] > 0.1) {
        edges.push({ source: nodes[j], target: nodes[i], weight: adj[i][j] });
      }
    }
  }

  return (
    <svg width="100%" height={size} viewBox={`0 0 ${size} ${size}`}>
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#34d399" opacity="0.8"/>
        </marker>
      </defs>
      {edges.map((edge, idx) => (
        <line
          key={`edge-${idx}`}
          x1={edge.source.x}
          y1={edge.source.y}
          x2={edge.target.x}
          y2={edge.target.y}
          stroke="#34d399"
          strokeWidth={Math.max(1, edge.weight * 3)}
          markerEnd="url(#arrowhead)"
          opacity="0.6"
        />
      ))}
      {nodes.map(node => (
        <g key={`node-${node.id}`}>
          <circle cx={node.x} cy={node.y} r="18" fill="#0b1a10" stroke="#34d399" strokeWidth="2" />
          <text x={node.x} y={node.y} textAnchor="middle" dy=".3em" fill="#ffffff" fontSize="12" fontWeight="600">
            Z{node.id}
          </text>
        </g>
      ))}
    </svg>
  );
};

export default function App() {
  const [systemBooted, setSystemBooted] = useState(false);
  const [config, setConfig] = useState({
    dataset: '3dident',
    epochs: 50,
    samples: 5000,
    lambda_mi: 1.0,
    lr: 0.001
  });

  const [status, setStatus] = useState({ is_training: false, current_epoch: 0, total_epochs: 50, history: [] });
  const [results, setResults] = useState(null);

  const pollStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/status');
      const data = await res.json();
      setStatus(data);
      if (data.is_training) {
        setTimeout(pollStatus, 1000);
      } else if (data.history.length > 0) {
        fetchResults();
      }
    } catch (e) {
      console.error(e);
      setTimeout(pollStatus, 2000);
    }
  };

  const fetchResults = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/results');
      const data = await res.json();
      if (data.metrics) setResults(data);
    } catch (e) { console.error(e); }
  };

  useEffect(() => {
    if (systemBooted) {
      pollStatus();
    }
  }, [systemBooted]);

  const handleTrain = async () => {
    setResults(null);
    try {
      await fetch('http://localhost:8000/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      pollStatus();
    } catch (e) { console.error(e); }
  };

  if (!systemBooted) {
    return (
      <div className="landing-page">
        <div className="landing-content">
          <h1 className="main-title">MICoRe</h1>
          <p className="subtitle">Identifiable Causal Representation Learning via Sparse Soft Interventions and Minimal Intervention Regularization</p>
          
          <div className="info-grid">
            <div className="info-card">
              <h3><BrainCircuit size={24}/> The Problem</h3>
              <p>
                Standard representation learning methods (like standard VAEs) often entangle latent spaces, making it impossible to map learned variables to actual physical causal factors without hard interventions or a known causal graph.
              </p>
            </div>
            
            <div className="info-card">
              <h3><Network size={24}/> The MICoRe Solution</h3>
              <p>
                MICoRe combines Identifiable Variational Autoencoders (iVAE) with continuous DAG learning (NOTEARS). It forces the network to learn disentangled representations by exploiting <strong>sparse soft interventions</strong> across different environments.
              </p>
            </div>
            
            <div className="info-card">
              <h3><Database size={24}/> Datasets</h3>
              <p>The system is evaluated on robust causal discovery benchmarks:</p>
              <ul>
                <li><strong>Causal3DIdent</strong>: Complex synthetic 3D scenes with latent causal factors like position, rotation, and color.</li>
                <li><strong>Pendulum</strong>: A physical dataset of coupled pendulums to test directional causal link recovery.</li>
              </ul>
            </div>
            
            <div className="info-card">
              <h3><ChartIcon size={24}/> Minimal Intervention Regularization</h3>
              <p>
                We introduce a novel {"$L_{MI}$"} sparsity penalty. It mathematically forces the causal mechanisms of most variables to remain invariant across environments, perfectly capturing the assumption that interventions in nature are typically sparse.
              </p>
            </div>
          </div>

          <div className="action-container">
            <button className="launch-btn" onClick={() => setSystemBooted(true)}>
              Launch Dashboard <ArrowRight size={20} style={{ display: 'inline', marginLeft: '8px', verticalAlign: 'middle' }} />
            </button>
          </div>
        </div>
      </div>
    );
  }

  const progress = status.total_epochs ? (status.current_epoch / status.total_epochs) * 100 : 0;
  const latestMetrics = status.history.length > 0 ? status.history[status.history.length - 1] : null;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: '#112a19', border: '1px solid rgba(52, 211, 153, 0.2)', padding: '16px', borderRadius: '12px', color:'#ffffff', boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)' }}>
          <p style={{ margin: '0 0 12px 0', fontWeight:600, color: '#34d399' }}>Epoch: {label}</p>
          {payload.map((entry, index) => (
            <p key={`item-${index}`} style={{ color: entry.color, margin: '6px 0', fontSize:'0.95rem', fontWeight: 500 }}>
              {entry.name}: {entry.value.toExponential ? entry.value.toExponential(2) : entry.value.toFixed(4)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar" style={{ borderRadius: 0, borderTop: 'none', borderBottom: 'none', borderLeft: 'none' }}>
        <div className="sidebar-header">
          <h1>MICoRe Dashboard</h1>
        </div>
        <div className="sidebar-content">
          <div className="form-group">
            <label className="form-label">Dataset Selection</label>
            <select 
              className="form-select" 
              value={config.dataset} 
              onChange={e => setConfig({...config, dataset: e.target.value})}
              disabled={status.is_training}
            >
              <option value="3dident">Synthetic 3D-Ident</option>
              <option value="pendulum">Coupled Pendulum</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Training Epochs</label>
            <input 
              type="number" 
              className="form-input" 
              value={config.epochs} 
              onChange={e => setConfig({...config, epochs: parseInt(e.target.value)})}
              disabled={status.is_training}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Samples per Environment</label>
            <input 
              type="number" 
              className="form-input" 
              value={config.samples} 
              onChange={e => setConfig({...config, samples: parseInt(e.target.value)})}
              disabled={status.is_training}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Minimal Intervention Weight (λ)</label>
            <input 
              type="number" 
              step="0.1"
              className="form-input" 
              value={config.lambda_mi} 
              onChange={e => setConfig({...config, lambda_mi: parseFloat(e.target.value)})}
              disabled={status.is_training}
            />
          </div>
          <button 
            className="btn btn-primary" 
            onClick={handleTrain}
            disabled={status.is_training}
          >
            <Play size={18} style={{ marginRight: '8px' }} />
            {status.is_training ? 'Training Model...' : 'Start Execution'}
          </button>
          
          {status.is_training && (
            <div className="progress-container">
              <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="header">
          <h2>Telemetry Overview</h2>
          <span className={`status-badge ${status.is_training ? 'training' : ''}`}>
            {status.is_training ? 'Active Training Phase' : 'System Ready'}
          </span>
        </div>

        {/* Metrics Grid */}
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Identifiability (MCC)</div>
            <div className="metric-value">{latestMetrics?.mcc ? latestMetrics.mcc.toFixed(4) : '0.0000'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">DAG Violation (h)</div>
            <div className="metric-value">{latestMetrics?.h ? latestMetrics.h.toExponential(2) : '0.00e+0'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Reconstruction Error</div>
            <div className="metric-value">{latestMetrics?.rec ? latestMetrics.rec.toFixed(4) : '0.0000'}</div>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="charts-grid">
          <div className="card">
            <div className="card-title"><Activity size={20} color="#2ecc71"/> Model Convergence</div>
            <div style={{ height: 260, marginTop: '10px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={status.history}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{fontSize: 12, fill: '#94b8a3'}} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="left" tick={{fontSize: 12, fill: '#94b8a3'}} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line yAxisId="left" type="monotone" dataKey="rec" stroke="#ffffff" strokeWidth={2} dot={false} name="Reconstruction" />
                  <Line yAxisId="left" type="monotone" dataKey="kl" stroke="#34d399" strokeWidth={2} dot={false} name="VAE KL" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div className="card">
            <div className="card-title"><BarChart3 size={20} color="#2ecc71"/> Causal Recovery Progress</div>
            <div style={{ height: 260, marginTop: '10px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={status.history}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{fontSize: 12, fill: '#94b8a3'}} tickLine={false} axisLine={false} />
                  <YAxis tick={{fontSize: 12, fill: '#94b8a3'}} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="mcc" stroke="#34d399" strokeWidth={2} dot={false} name="MCC" />
                  <Line type="monotone" dataKey="shd" stroke="#ef4444" strokeWidth={2} dot={false} name="SHD" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Results Section */}
        {results && (
          <div className="charts-grid">
            <div className="card">
              <div className="card-title"><Network size={20} color="#2ecc71"/> Final Causal Graph (DAG)</div>
              <div style={{display:'flex', justifyContent:'center', alignItems:'center', height:'100%'}}>
                <CausalGraph adj={results.adj} />
              </div>
            </div>
            
            <div className="card">
              <div className="card-title"><Database size={20} color="#34d399"/> DCI Evaluation Matrix</div>
              <div style={{ padding: '20px 0', fontSize:'1.1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px', borderBottom: '1px solid rgba(52, 211, 153, 0.1)', paddingBottom: '12px' }}>
                  <span style={{ color: '#94b8a3' }}>Disentanglement</span>
                  <span style={{ fontWeight: 600, color: '#ffffff' }}>{results.metrics.dci.disentanglement.toFixed(4)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px', borderBottom: '1px solid rgba(52, 211, 153, 0.1)', paddingBottom: '12px' }}>
                  <span style={{ color: '#94b8a3' }}>Completeness</span>
                  <span style={{ fontWeight: 600, color: '#ffffff' }}>{results.metrics.dci.completeness.toFixed(4)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94b8a3' }}>Informativeness</span>
                  <span style={{ fontWeight: 600, color: '#ffffff' }}>{results.metrics.dci.informativeness.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
