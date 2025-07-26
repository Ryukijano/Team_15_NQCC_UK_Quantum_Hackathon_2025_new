# Born-Heisenberg Hamiltonian Architecture Flowchart

```mermaid
flowchart TD
    A[Start: sqd_heisenberg.py] --> B[NickelHydrogenSQD Class]
    B --> C{hamiltonian_type = 'heisenberg'?}
    C -->|Yes| D[build_born_heisenberg_hamiltonian]
    C -->|No| E[Other Hamiltonian Types]
    
    D --> F[Heisenberg XXZ Model Construction]
    F --> G[Exchange Coupling J]
    F --> H[Anisotropy Parameter Δ]
    F --> I[Local Field h]
    
    G --> J[Generate Pauli Strings]
    H --> J
    I --> J
    
    J --> K[XX Terms: 'XXII', 'IXXI', 'IIXX']
    J --> L[YY Terms: 'YYII', 'IYYI', 'IIYY']
    J --> M[ZZ Terms: 'ZZII', 'IZZI', 'IIZZ']
    J --> N[Z Field Terms: 'ZIII', 'IZII', 'IIZI', 'IIIZ']
    
    K --> O[SparsePauliOp Hamiltonian]
    L --> O
    M --> O
    N --> O
    
    O --> P[SampleBasedQuantumDiagonalization]
    
    P --> Q[Initial State Preparation]
    Q --> R[Product State |ψ₀⟩]
    
    R --> S[Krylov Subspace Construction]
    S --> T[Time Evolution: |ψₖ⟩ = e⁻ⁱᵏᴴᵀᵗ|ψ₀⟩]
    
    T --> U[Quantum Sampling]
    U --> V[Measure Krylov States]
    V --> W[Bitstring Samples]
    
    W --> X[Configuration Recovery]
    X --> Y[Filter Physical Constraints]
    Y --> Z[Total Magnetization Filter]
    
    Z --> AA[Subspace Projection]
    AA --> BB[Project Hamiltonian into Sampled Subspace]
    
    BB --> CC[Classical Diagonalization]
    CC --> DD[Eigenvalues & Eigenvectors]
    
    DD --> EE[Energy Spectrum]
    EE --> FF[Visualization & Analysis]
    
    %% Mapping Information
    subgraph "Spin-to-Qubit Mapping"
        GG[Spin-1/2 Site] --> HH[1 Qubit]
        II[Pauli Operators X,Y,Z] --> JJ[Direct Qubit Operations]
    end
    
    %% Hamiltonian Form
    subgraph "Hamiltonian Form"
        KK[H = J∑ᵢ(XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ΔZᵢZᵢ₊₁) + h∑ᵢZᵢ]
    end
    
    %% Key Parameters
    subgraph "Default Parameters"
        LL[J = 1.0]
        MM[Δ = 1.0]
        NN[h = 0.0]
    end
    
    %% Styling
    classDef startEnd fill:#e1f5fe
    classDef process fill:#f3e5f5
    classDef decision fill:#fff3e0
    classDef data fill:#e8f5e8
    classDef mapping fill:#fce4ec
    
    class A,FF startEnd
    class B,D,F,P,Q,S,U,X,AA,CC process
    class C decision
    class G,H,I,J,K,L,M,N,O,R,T,V,W,Y,BB,DD,EE data
    class GG,HH,II,JJ mapping
```

## Architecture Components

### 1. **Hamiltonian Construction**
- **Model**: Heisenberg XXZ chain
- **Mapping**: Direct spin-to-qubit (1 spin = 1 qubit)
- **Terms**: XX, YY, ZZ interactions + optional Z field
- **Boundary**: Open boundary conditions (no periodic wrap-around)

### 2. **Quantum-Classical Hybrid Workflow**
- **Quantum Part**: Time evolution and sampling
- **Classical Part**: Subspace projection and diagonalization
- **Interface**: Sample-based quantum diagonalization

### 3. **Key Features**
- **No Fermion Mapping**: Direct spin model → qubit mapping
- **Scalable**: Linear qubit scaling with system size
- **Flexible**: Configurable coupling constants and field strength
- **Physical**: Respects spin-1/2 constraints

### 4. **Data Flow**
1. **Input**: System parameters (J, Δ, h, N qubits)
2. **Quantum Processing**: Time evolution and sampling
3. **Classical Processing**: Subspace diagonalization
4. **Output**: Energy spectrum and eigenstates

### 5. **Optimization Opportunities**
- **Parallelization**: Independent time evolution steps
- **Sampling**: Parallel quantum measurements
- **Diagonalization**: Efficient sparse matrix methods
- **Memory**: Sparse representation of Pauli operators 
