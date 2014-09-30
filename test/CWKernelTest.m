A =           [ 0, 1, 0;... 
                1, 0, 1;... 
                0, 1, 0];

train_ind = [3]';
observed_labels = [1]';
test_ind = [1,2]';
num_classes = 2;
walk_length = 5;

[K_train, K_test] = CWKernel(A, train_ind, observed_labels, test_ind, num_classes, walk_length, 'alpha',0);
save -ascii 'test/data/Solution0_train.txt' K_train
save -ascii 'test/data/Solution0_test.txt' K_test

A =           [ 0, 1, 0, 1, 0;... 
                1, 0, 1, 1, 0;... 
                0, 1, 0, 0, 1;...
                1, 1, 0, 0, 0;...
                0, 0, 1, 0, 0];

train_ind = [1,3,4]';
observed_labels = [1, 1, 2]';
test_ind = [2,5]';
num_classes = 2;
walk_length = 5;

[K_train, K_test] = CWKernel(A, train_ind, observed_labels, test_ind, num_classes, walk_length, 'alpha', 1);
save -ascii 'test/data/Solution1_train.txt' K_train
save -ascii 'test/data/Solution1_test.txt' K_test

A =           [ 0, 1, 0, 1, 0;... 
                1, 0, 1, 1, 0;... 
                0, 1, 0, 0, 1;...
                1, 1, 0, 0, 0;...
                0, 0, 1, 0, 0];

train_ind = [1,3,4]';
observed_labels = [1, 1, 2]';
test_ind = [2,5]';
num_classes = 2;
walk_length = 5;

[K_train, K_test] = CWKernel(A, train_ind, observed_labels, test_ind, num_classes, walk_length, 'alpha', 0.5);
save -ascii 'test/data/Solution2_train.txt' K_train
save -ascii 'test/data/Solution2_test.txt' K_test


load('test/data/citeseer.mat')
X_train = double(X_train)+1;
X_test = double(X_test)+1;
labels = double(labels(X_train)) +1;
A = double(A);
[K_train, K_test] = CWKernel(A, X_train, labels, X_test, 6, 6, 'alpha', 0.5);
save -ascii 'test/data/Solution3_train.txt' K_train
save -ascii 'test/data/Solution3_test.txt' K_test
