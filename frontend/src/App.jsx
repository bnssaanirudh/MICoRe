import { useState, useEffect, useRef } from 'react'
import anime from 'animejs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Play, Activity, Network, BarChart3, Database, BrainCircuit, Code, Cpu, ShieldAlert, Terminal } from 'lucide-react'

// Animated Background Component (Anime.js signature effect)
const AnimeBackground = () => {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    // Clear previous if re-rendered
    container.innerHTML = '';
    
    // Calculate grid size
    const elementSize = 44; // 40px width + 4px margin
    const columns = Math.floor(window.innerWidth / elementSize);
    const rows = Math.floor(window.innerHeight / elementSize);
    const totalElements = columns * rows;

    // Create DOM elements
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < totalElements; i++) {
      const el = document.createElement('div');
      el.classList.add('grid-element');
      fragment.appendChild(el);
    }
    container.appendChild(fragment);

    // Run anime.js animation
    anime({
      targets: '.grid-element',
      scale: [
        {value: 0.1, easing: 'easeOutSine', duration: 500},
        {value: 1, easing: 'easeInOutQuad', duration: 1200}
      ],
      delay: anime.stagger(200, {grid: [columns, rows], from: 'center'}),
      loop: true,
      direction: 'alternate',
      backgroundColor: ['#222222', '#FF4B4B', '#222222']
    });

    return () => {
      anime.remove('.grid-element');
    };
  }, []);

  return <div className="anime-background" ref={containerRef}></div>;
};

// SVG Graph Component for Adjacency Matrix (Brutalist Theme)
const CausalGraph = ({ adj }) => {
  if (!adj || adj.length === 0) return <div className="placeholder" style={{color: 'var(--text-muted)', textAlign:'center', marginTop:'40px', fontWeight:900}}>NO GRAPH DATA</div>;
  const size = 300;
  const center = size / 2;
  const radius = 100;
  const n = adj.length;

  const nodes = Array.from({ length: n }).map((_, i) => {
    const angle = (i * 2 * Math.PI) / n - Math.PI / 2;
    return { id: i, x: center + radius * Math.cos(angle), y: center + radius * Math.sin(angle) };
  });

  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (adj[i][j] > 0.1) edges.push({ source: nodes[j], target: nodes[i], weight: adj[i][j] });
    }
  }

  return (
    <svg width="100%" height={size} viewBox={`0 0 ${size} ${size}`}>
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#FF4B4B" />
        </marker>
      </defs>
      {edges.map((edge, idx) => (
        <line
          key={`edge-${idx}`} x1={edge.source.x} y1={edge.source.y} x2={edge.target.x} y2={edge.target.y}
          stroke="#FF4B4B" strokeWidth={Math.max(1, edge.weight * 3)} markerEnd="url(#arrowhead)"
        />
      ))}
      {nodes.map(node => (
        <g key={`node-${node.id}`}>
          <rect x={node.x - 20} y={node.y - 20} width="40" height="40" fill="#000" stroke="#FFF" strokeWidth="2" />
          <text x={node.x} y={node.y} textAnchor="middle" dy=".35em" fill="#fff" fontSize="14" fontWeight="900" fontFamily="Inter">Z{node.id}</text>
        </g>
      ))}
    </svg>
  );
};

export default function App() {
  const [showDashboard, setShowDashboard] = useState(false);
  const [config, setConfig] = useState({ dataset: '3dident', epochs: 50, samples: 5000, lambda_mi: 1.0, lr: 0.001 });
  const [status, setStatus] = useState({ is_training: false, current_epoch: 0, total_epochs: 50, history: [] });
  const [results, setResults] = useState(null);

  const pollStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/status');
      const data = await res.json();
      setStatus(data);
      if (data.is_training) setTimeout(pollStatus, 1000);
      else if (data.history.length > 0) fetchResults();
    } catch (e) {
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

  useEffect(() => { if (showDashboard) pollStatus(); }, [showDashboard]);

  const handleTrain = async () => {
    setResults(null);
    try {
      await fetch('http://localhost:8000/api/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config)
      });
      pollStatus();
    } catch (e) { console.error(e); }
  };

  if (!showDashboard) {
    return (
      <div className="website-container">
        {/* Navigation */}
        <nav className="navbar">
          <div className="nav-brand">MIC<span>oRe</span></div>
          <div className="nav-links">
            <a href="#features" className="nav-link">Features</a>
            <a href="#architecture" className="nav-link">Architecture</a>
          </div>
          <button className="nav-cta" onClick={() => setShowDashboard(true)}>Launch Platform</button>
        </nav>

        {/* Hero Section */}
        <section className="hero-section">
          <AnimeBackground />
          <div className="hero-content">
            <h1 className="hero-title">CAUSAL<br/>DISCOVERY</h1>
            <p className="hero-subtitle">Identifiable Representation Learning</p>
            <div className="hero-buttons">
              <button className="btn-brutal primary" onClick={() => setShowDashboard(true)}>Start Telemetry</button>
              <button className="btn-brutal" style={{marginLeft: '20px'}}>Documentation</button>
            </div>
          </div>
        </section>

        {/* Stats / Marquee Bar */}
        <div className="stats-bar">
          <div className="stat-block">
            <div className="stat-value">1D</div>
            <div className="stat-label">Wasserstein Loss</div>
          </div>
          <div className="stat-block">
            <div className="stat-value">0.0</div>
            <div className="stat-label">DAG Violations</div>
          </div>
          <div className="stat-block">
            <div className="stat-value">MCC</div>
            <div className="stat-label">Identifiability Guarantee</div>
          </div>
        </div>

        {/* Features Section */}
        <section id="features" className="section">
          <div className="section-header">
            <h2 className="section-title">SYSTEM <span>CAPABILITIES</span></h2>
          </div>
          
          <div className="brutal-grid">
            <div className="brutal-card">
              <h3>DISENTANGLEMENT</h3>
              <p>Map independent priors onto exogenous noise variables. Guarantee mathematically tractable likelihood estimation.</p>
            </div>
            <div className="brutal-card">
              <h3>NOTEARS DAG</h3>
              <p>Continuous causal adjacency matrix W constrained strictly to a Directed Acyclic Graph. Jacobian = 1.</p>
            </div>
            <div className="brutal-card">
              <h3>SPARSE SHIFT</h3>
              <p>Minimal Intervention Loss (L_MI) forces invariant causal mechanisms across environments effortlessly.</p>
            </div>
          </div>
        </section>

        {/* Code Deep Dive */}
        <section id="architecture" className="code-section">
          <div className="code-container">
            <div className="code-header">
              <span>micore_engine/loss.py</span>
              <span>Python</span>
            </div>
            <div className="code-body">
              def get_intervention_loss(self, z_0, z_u, lambda_mi=1.0):<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;# Extract exogenous samples<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;eps_0 = self.get_exogenous(z_0)<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;eps_u = self.get_exogenous(z_u)<br/><br/>
              &nbsp;&nbsp;&nbsp;&nbsp;# 1D Wasserstein exactly identifies sparse shift<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;w1_dist = torch.mean(torch.abs(torch.sort(eps_u, dim=0)[0] - torch.sort(eps_0, dim=0)[0]))<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;return lambda_mi * w1_dist<br/>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="footer">
          <div className="footer-col">
            <h2 style={{fontSize: '2rem', marginBottom: '10px'}}>MIC<span style={{color: 'var(--accent-coral)'}}>oRe</span></h2>
            <p style={{color: 'var(--text-muted)', maxWidth: '300px'}}>Open Source Identifiable Causal Representation Learning Framework.</p>
          </div>
          <div className="footer-col">
            <h4>DEVELOPMENT</h4>
            <ul>
              <li><a href="#">Documentation</a></li>
              <li><a href="#">API Reference</a></li>
              <li><a href="#">Benchmarks</a></li>
            </ul>
          </div>
          <div className="footer-col">
            <h4>SOCIAL</h4>
            <ul>
              <li><a href="#"><Terminal size={14}/> GitHub</a></li>
              <li><a href="#">Discord</a></li>
              <li><a href="#">Paper</a></li>
            </ul>
          </div>
        </footer>
      </div>
    );
  }

  // --- DASHBOARD UI ---
  const progress = status.total_epochs ? (status.current_epoch / status.total_epochs) * 100 : 0;
  const latestMetrics = status.history.length > 0 ? status.history[status.history.length - 1] : null;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: '#000', border: '1px solid #333', padding: '16px', color:'#ffffff' }}>
          <p style={{ margin: '0 0 12px 0', fontWeight:900, color: '#FF4B4B' }}>EPOCH {label}</p>
          {payload.map((entry, index) => (
            <p key={`item-${index}`} style={{ color: entry.color, margin: '6px 0', fontSize:'0.9rem', fontWeight: 600, fontFamily: 'Fira Code, monospace' }}>
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
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>MIC<span>oRe</span></h1>
        </div>
        <div className="sidebar-content">
          <div className="form-group">
            <label className="form-label">DATASET ENVIRONMENT</label>
            <select className="form-select" value={config.dataset} onChange={e => setConfig({...config, dataset: e.target.value})} disabled={status.is_training}>
              <option value="3dident">SYNTHETIC 3D-IDENT</option>
              <option value="pendulum">COUPLED PENDULUM</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">TRAINING EPOCHS</label>
            <input type="number" className="form-input" value={config.epochs} onChange={e => setConfig({...config, epochs: parseInt(e.target.value)})} disabled={status.is_training}/>
          </div>
          <div className="form-group">
            <label className="form-label">SAMPLES/ENV</label>
            <input type="number" className="form-input" value={config.samples} onChange={e => setConfig({...config, samples: parseInt(e.target.value)})} disabled={status.is_training}/>
          </div>
          <div className="form-group">
            <label className="form-label">WASSERSTEIN PENALTY (λ)</label>
            <input type="number" step="0.1" className="form-input" value={config.lambda_mi} onChange={e => setConfig({...config, lambda_mi: parseFloat(e.target.value)})} disabled={status.is_training}/>
          </div>
          
          <button className="btn-dashboard" onClick={handleTrain} disabled={status.is_training}>
            <Play size={18} fill={status.is_training ? "none" : "#fff"}/>
            {status.is_training ? 'SYSTEM TRAINING...' : 'INITIALIZE SEQUENCE'}
          </button>
          
          {status.is_training && (
            <div className="progress-line">
              <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>
          )}
        </div>
      </div>

      <div className="main-content">
        <div className="header-dash">
          <h2>TELEMETRY<br/>OVERVIEW</h2>
          <span className={`status-badge ${status.is_training ? 'training' : ''}`}>
            {status.is_training ? 'TRAINING ACTIVE' : 'SYSTEM READY'}
          </span>
        </div>

        <div className="metrics-row">
          <div className="metric-box">
            <div className="label">Identifiability (MCC)</div>
            <div className="value" style={{color: '#FF4B4B'}}>{latestMetrics?.mcc ? latestMetrics.mcc.toFixed(4) : '0.0000'}</div>
          </div>
          <div className="metric-box">
            <div className="label">DAG Violation (h)</div>
            <div className="value">{latestMetrics?.h ? latestMetrics.h.toExponential(2) : '0.00E+0'}</div>
          </div>
          <div className="metric-box">
            <div className="label">Reconstruction Error</div>
            <div className="value">{latestMetrics?.rec ? latestMetrics.rec.toFixed(4) : '0.0000'}</div>
          </div>
        </div>

        <div className="charts-row">
          <div className="chart-panel">
            <div className="chart-header"><Activity size={20} color="#FF4B4B"/> CONVERGENCE KINETICS</div>
            <div className="chart-body">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={status.history}>
                  <CartesianGrid strokeDasharray="0" vertical={false} />
                  <XAxis dataKey="epoch" tick={{fontSize: 12, fill: '#888', fontFamily: 'Fira Code'}} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="left" tick={{fontSize: 12, fill: '#888', fontFamily: 'Fira Code'}} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line yAxisId="left" type="stepAfter" dataKey="rec" stroke="#ffffff" strokeWidth={2} dot={false} name="L_REC" />
                  <Line yAxisId="left" type="stepAfter" dataKey="kl" stroke="#FF4B4B" strokeWidth={2} dot={false} name="L_VAE" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div className="chart-panel">
            <div className="chart-header"><BarChart3 size={20} color="#FF4B4B"/> GRAPH RECOVERY</div>
            <div className="chart-body">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={status.history}>
                  <CartesianGrid strokeDasharray="0" vertical={false} />
                  <XAxis dataKey="epoch" tick={{fontSize: 12, fill: '#888', fontFamily: 'Fira Code'}} tickLine={false} axisLine={false} />
                  <YAxis tick={{fontSize: 12, fill: '#888', fontFamily: 'Fira Code'}} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="stepAfter" dataKey="mcc" stroke="#FF4B4B" strokeWidth={2} dot={false} name="MCC" />
                  <Line type="stepAfter" dataKey="shd" stroke="#18FFB5" strokeWidth={2} dot={false} name="SHD" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {results && (
          <div className="charts-row">
            <div className="chart-panel">
              <div className="chart-header"><Network size={20} color="#FF4B4B"/> CAUSAL STRUCTURE (DAG)</div>
              <div style={{display:'flex', justifyContent:'center', alignItems:'center', height:'300px'}}>
                <CausalGraph adj={results.adj} />
              </div>
            </div>
            
            <div className="chart-panel">
              <div className="chart-header"><Database size={20} color="#FF4B4B"/> DCI METRICS</div>
              <div style={{ padding: '40px', fontSize:'1.2rem', fontFamily: 'Fira Code' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px', borderBottom: '1px solid #333', paddingBottom: '12px' }}>
                  <span style={{ color: '#888' }}>DISENTANGLEMENT</span>
                  <span style={{ fontWeight: 900, color: '#fff' }}>{results.metrics.dci.disentanglement.toFixed(4)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px', borderBottom: '1px solid #333', paddingBottom: '12px' }}>
                  <span style={{ color: '#888' }}>COMPLETENESS</span>
                  <span style={{ fontWeight: 900, color: '#fff' }}>{results.metrics.dci.completeness.toFixed(4)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#888' }}>INFORMATIVENESS</span>
                  <span style={{ fontWeight: 900, color: '#fff' }}>{results.metrics.dci.informativeness.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
