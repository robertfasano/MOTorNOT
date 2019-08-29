import numpy as np
import plotly.graph_objs as go
from MOTorNOT.plotting import plot_2D, plot_phase_space_force

def atoms_to_arrays(atoms):
    x = []
    v = []
    for atom in atoms:
        x = np.append(x, atom.x)
        v = np.append(v, atom.v)

    return x, v

def phase_space_trajectories(mot, atoms, axis='x'):
    xatoms, vatoms = atoms_to_arrays(atoms)

    surf = plot_phase_space_force(mot.acceleration, axis, limits=[(xatoms.min(), xatoms.max()), (1.5*vatoms.min(), 1.5*vatoms.max())], numpoints=100)


    index = axis
    vindex = f'v{index}'
    traces = [surf]
    for atom in atoms:
        traces.append(go.Scatter(x=atom.x[index], y=atom.v[vindex], line=dict(color='#ffffff')))
        traces.append(go.Scatter(x=[atom.x.iloc[-1][index]], y=[atom.v.iloc[-1][vindex]], line=dict(color='#000000')))
    fig = go.Figure(traces)
    fig.update_layout(showlegend=False)
    fig.show()


def trajectories(mot, atoms, plane='xy'):
    fig = plot_2D(mot.acceleration, plane, limits=[(-15e-3, 15e-3), (-15e-3, 15e-3)], numpoints=30)

    for atom in atoms:
        fig.add_trace(go.Scatter(x=atom.x[plane[0]], y=atom.x[plane[1]], line=dict(color='#ffffff')))
        fig.add_trace(go.Scatter(x=[atom.x.iloc[-1][plane[0]]], y=[atom.x.iloc[-1][plane[1]]], line=dict(color='#000000')))
    fig.update_layout(showlegend=False)

    fig.show()
