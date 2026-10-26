"""Targeted numerical and behavioral checks, not a production API conformance suite."""
from pathlib import Path
import contextlib
import importlib.util
import io
import os
import sys
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in sorted(ROOT.glob("[0-9][0-9]_*")):
    sys.path.append(str(folder))


def load(folder, stem):
    path = ROOT / folder / (stem + ".py")
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


knn = load("02_knn_from_scratch", "knn_classification")
km = load("03_kmeans_from_scratch", "kmeans_clustering")
linear = load("04_linear_regression_from_scratch", "linear_regression")
logistic = load("05_logistic_regression_from_scratch", "logistic_regression")
tree = load("06_decision_tree_from_scratch", "decision_tree")
forest = load("07_random_forest_from_scratch", "random_forest")
ensemble = load("08_ensemble_learning", "ensemble_learning")
gb = load("08_ensemble_learning", "gradient_boosting_regression")
nb = load("09_gaussian_naive_bayes_from_scratch", "gaussian_naive_bayes")
svm = load("10_svm_form_scratch", "svm")
pca = load("11_pca_from_scratch", "pca")


class AlgorithmChecks(unittest.TestCase):
    def test_knn_distance_and_ties(self):
        X = np.array([[0., 0.], [3., 4.], [-3., -4.]])
        y = np.array([1, 0, 0])
        self.assertEqual(knn.predict_one(X, y, np.array([.1, .1]), 1), 1)
        # Equal distances preserve training order; vote ties choose smaller class.
        self.assertEqual(knn.predict_one(X[1:], np.array([1, 0]), np.zeros(2), 1), 1)
        self.assertEqual(knn.predict_one(X[1:], np.array([1, 0]), np.zeros(2), 2), 0)

    def test_knn_invalid_k(self):
        for k in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                knn.predict_one(np.zeros((3, 2)), np.zeros(3, dtype=int), np.zeros(2), k)

    def test_kmeans_two_separated_groups(self):
        X = np.array([[-4., 0.], [-2., 0.], [2., 0.], [4., 0.]])
        centers, ids, _ = km.fit_kmeans(X, 2, 42, 50, 1e-10)
        np.testing.assert_allclose(np.sort(centers[:, 0]), [-3., 3.])
        np.testing.assert_array_equal(ids, km.assign_cluster(X, centers))
        self.assertAlmostEqual(km.compute_inertia(X, ids, centers), 4.)

    def test_kmeans_empty_cluster_keeps_center(self):
        centers = np.array([[0., 0.], [10., 10.]])
        result = km.update_centroids(np.array([[1., 1.], [3., 3.]]), np.array([0, 0]), centers)
        np.testing.assert_allclose(result, [[2., 2.], [10., 10.]])

    def test_linear_gradient_finite_difference(self):
        x = np.array([-1., .3, 2.]); y = np.array([0., 2., 5.]); w, b = .4, .7
        actual = linear.compute_gradients(x, y, linear.predict(x, w, b))
        loss = lambda w, b: linear.compute_mse(y, linear.predict(x, w, b))
        eps = 1e-6
        expected = [(loss(w+eps,b)-loss(w-eps,b))/(2*eps), (loss(w,b+eps)-loss(w,b-eps))/(2*eps)]
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_linear_learns_exact_rule(self):
        x = np.linspace(-2, 2, 30); y = 3*x+2
        w, b, history = linear.fit_linear_regression(x, y, 0., 0., .1, 300)
        np.testing.assert_allclose([w, b], [3., 2.], atol=1e-8)
        self.assertLess(history[-1], history[0])
        self.assertEqual(len(history), 301)

    def test_sigmoid_extreme_scores(self):
        with np.errstate(over="raise", invalid="raise"):
            result = logistic.sigmoid(np.array([-1000., 0., 1000.]))
        np.testing.assert_allclose(result, [0., .5, 1.])

    def test_logistic_gradient_finite_difference(self):
        x = np.array([-1., .3, 2.]); y = np.array([0, 1, 1]); w, b = .4, .7
        actual = logistic.compute_gradients(x, y, logistic.predict_proba(x,w,b))
        loss = lambda w,b: logistic.compute_bce(y, logistic.predict_proba(x,w,b))
        eps = 1e-6
        expected = [(loss(w+eps,b)-loss(w-eps,b))/(2*eps), (loss(w,b+eps)-loss(w,b-eps))/(2*eps)]
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_gini_and_threshold(self):
        self.assertEqual(tree.compute_gini(np.array([0,0,1,1])), .5)
        threshold, score = tree.find_best_threshold(np.array([1.,2.,3.,4.]),np.array([0,0,1,1]))
        self.assertEqual(threshold, 2.5); self.assertEqual(score, 0.)

    def test_tree_stops_and_predicts(self):
        X = np.array([[1.],[2.],[3.],[4.]]); y = np.array([0,0,1,1])
        fitted = tree.build_tree(X,y,2)
        np.testing.assert_array_equal(tree.predict(np.array([[2.5],[3.]]),fitted),[0,1])
        self.assertTrue(tree.build_tree(X,y,0)["is_leaf"])
        self.assertTrue(tree.build_tree(X,y,3,min_samples_split=5)["is_leaf"])
        self.assertTrue(tree.build_tree(np.ones_like(X),y,3)["is_leaf"])

    def test_documented_zero_gain_tree_limitation(self):
        X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]);y=np.array([0,1,1,0])
        self.assertTrue(tree.build_tree(X,y,2)["is_leaf"])
        # L06 allows a zero-gain first split and can subsequently solve XOR.
        fitted=ensemble.build_tree(X,y,2,2)
        np.testing.assert_array_equal(tree.predict(X,fitted),y)

    def test_bootstrap_preserves_pairs(self):
        X=np.arange(20).reshape(10,2); y=np.arange(10)
        a,b=forest.bootstrap_sampling(X,y,np.random.default_rng(42))
        np.testing.assert_array_equal(a, X[b])
        self.assertEqual(a.shape,X.shape)
        self.assertLess(len(np.unique(b)),len(y))

    def test_forest_seed_and_prefix(self):
        X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); y=np.array([0,0,1,1])
        args=dict(max_depth=3,min_samples_split=2,max_features=1,seed=42)
        first=forest.fit_random_forest(X,y,n_estimators=2,**args)
        again=forest.fit_random_forest(X,y,n_estimators=3,**args)
        self.assertEqual(first,again[:2])
        np.testing.assert_array_equal(forest.predict_random_forest(X,first),forest.predict_random_forest(X,again[:2]))

    def test_forest_rejects_empty_and_invalid_features(self):
        with self.assertRaises(ValueError): forest.predict_random_forest(np.zeros((1,2)),[])
        with self.assertRaises(ValueError): forest.select_feature_indices(2,True,np.random.default_rng(0))
        with self.assertRaises(ValueError):
            forest.fit_random_forest(np.ones((2,2)),np.array([0,1]),0,2,2,1,42)

    def test_weighted_stump_uses_weighted_majority(self):
        X=np.ones((3,1));y=np.array([0,0,1]);weights=np.array([.1,.1,.8])
        stump=ensemble.fit_weighted_stump(X,y,weights)
        np.testing.assert_array_equal(tree.predict(X,stump),[1,1,1])
        self.assertAlmostEqual(ensemble.compute_weighted_gini(y,weights),.32)

    def test_adaboost_update_and_perfect_learner(self):
        X=np.array([[0.],[1.],[2.],[3.]]); y=np.array([0,0,1,1])
        with contextlib.redirect_stdout(io.StringIO()):stumps,alphas,errors=ensemble.fit_adaboost(X,y,5)
        self.assertEqual(len(stumps),1);self.assertEqual(errors[0],0)
        np.testing.assert_array_equal(ensemble.predict_adaboost(X,stumps,alphas),y)
        # Independent check of the stated binary weight update property.
        e=.25; alpha=.5*np.log((1-e)/e)
        wrong=e*np.exp(alpha);correct=(1-e)*np.exp(-alpha)
        self.assertAlmostEqual(wrong/(wrong+correct),.5)
        with self.assertRaises(ValueError):ensemble.fit_adaboost(X,np.array([0,0,2,2]),5)
        with self.assertRaises(ValueError):ensemble.predict_adaboost(X,stumps,[])

    def test_gradient_boosting_residual_correction(self):
        x=np.array([0.,1.,2.,3.]);y=np.array([0.,0.,8.,8.])
        initial,stumps,history=gb.fit_gradient_boosting(x,y,1.,1)
        self.assertEqual(initial,4.)
        np.testing.assert_allclose(gb.predict_gradient_boosting(x,initial,stumps,1.),y)
        self.assertAlmostEqual(history[-1],0.)

    def test_regression_stump_constant_feature(self):
        x=np.ones(3);targets=np.array([1.,2.,6.])
        stump=gb.fit_regression_stump(x,targets)
        self.assertIsNone(stump["threshold"])
        np.testing.assert_allclose(gb.predict_regression_stump(x,stump),3.)

    def test_naive_bayes_numeric_posterior_and_class_mapping(self):
        X=np.array([[30.],[32.],[34.],[36.],[40.],[44.]])
        model=nb.fit_gaussian_naive_bayes_multifeature(X,np.array([2,2,2,2,7,7]))
        pred,p=nb.predict_one_gaussian_naive_bayes_multifeature(np.array([40.]),model)
        self.assertEqual(pred,7)
        np.testing.assert_allclose(p,[.02149039324638876,.9785096067536113])

    def test_naive_bayes_constant_variance_and_invalid_floor(self):
        X=np.array([[0.,1.],[0.,1.],[2.,1.],[2.,1.]])
        model=nb.fit_gaussian_naive_bayes_multifeature(X,np.array([0,0,1,1]))
        pred,p=nb.predict_gaussian_naive_bayes_multifeature(np.array([[1000.,1.]]),model)
        self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p.sum(1),1.)
        with self.assertRaises(ValueError):nb.fit_gaussian_naive_bayes_multifeature(X,np.array([0,0,1,1]),0.)

    def test_svm_gradient_finite_difference_away_from_kink(self):
        X=np.array([[-.5,-.2],[.6,.3]]);y=np.array([-1,1]);w=np.array([.1,.2]);b=.05;C=1.
        dw,db=svm.compute_svm_gradients(X,y,w,b,C);eps=1e-6
        expected=[]
        for j in range(2):
            offset=np.eye(2)[j]*eps
            expected.append((svm.compute_svm_loss(X,y,w+offset,b,C)[0]-svm.compute_svm_loss(X,y,w-offset,b,C)[0])/(2*eps))
        expected_b=(svm.compute_svm_loss(X,y,w,b+eps,C)[0]-svm.compute_svm_loss(X,y,w,b-eps,C)[0])/(2*eps)
        np.testing.assert_allclose(dw,expected,atol=1e-8);self.assertAlmostEqual(db,expected_b)

    def test_svm_inactive_and_kink_convention(self):
        X=np.array([[-2.],[2.]]);y=np.array([-1,1]);w=np.array([.5])
        dw,db=svm.compute_svm_gradients(X,y,w,0.,1.)
        np.testing.assert_allclose(dw,w);self.assertEqual(db,0.)
        total,penalty,hinge=svm.compute_svm_loss(X,y,w,0.,1.)
        self.assertEqual(hinge,0.);self.assertEqual(total,penalty)

    def test_pca_known_variance_and_reconstruction(self):
        X=np.array([[-1.5,-2.5],[-1.5,-.5],[0.,0.],[.5,1.5],[2.5,1.5]])
        model=pca.fit_pca(X,1);projected=pca.transform_pca(X,model)
        restored=pca.inverse_transform_pca(projected,model)
        np.testing.assert_allclose(model["explained_variance"],[4.])
        np.testing.assert_allclose(model["explained_variance_ratio"],[4/4.4])
        self.assertAlmostEqual(np.mean((X-restored)**2),.2)

    def test_pca_mean_full_roundtrip_and_new_batch(self):
        train=np.array([[-2.,-1.],[-1.,-2.],[1.,2.],[2.,1.]])+[10.,5.]
        full=pca.fit_pca(train,2);query=np.array([[100.,200.],[4.,8.]])
        np.testing.assert_allclose(pca.inverse_transform_pca(pca.transform_pca(query,full),full),query)
        # A singleton must use training mean, not re-center itself to zero.
        one=pca.fit_pca(train,1)
        manual=(query[:1]-train.mean(0))@one["components"]
        np.testing.assert_allclose(pca.transform_pca(query[:1],one),manual)
        self.assertGreater(np.linalg.norm(manual),1.)
        np.testing.assert_allclose(one["mean"],train.mean(0))

    def test_pca_constant_and_invalid_dimension(self):
        model=pca.fit_pca(np.ones((4,2)),1)
        np.testing.assert_array_equal(model["explained_variance_ratio"],[0.])
        with self.assertRaises(ValueError):pca.fit_pca(np.ones((4,2)),True)
        with self.assertRaises(ValueError):pca.transform_pca(np.ones((2,3)),model)


if __name__ == "__main__":
    unittest.main()
