from umap.parametric_umap import ParametricUMAP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

umap = Pipeline([('scaling', StandardScaler()), ('umap', ParametricUMAP(n_components=2))])
#embeddings_2ds = umap.fit_transform(all_embeddings)
#umap.fit(all_embeddings)
#umap.save('umap_model_both.saved')
# Save the PCA model
#joblib.dump(umap, 'umap_model_both.pkl')
#joblib.dump(umap, 'umap_model_both.pkl')
