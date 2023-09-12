#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <string>
#include <fstream>

using namespace std;
#define H_EXPRESSION (count_1_ * 3)

int N;

class Node {
public:
    int x;
    int y;
    int s;
    int G;//G
    int H;
    int F;
    int table[12][12]{};
    Node *parent;

    Node(Node *pNode, int change_x, int change_y, int change_type) {
        parent = pNode;
        x = change_x;
        y = change_y;
        s = change_type;
        if (parent) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    table[i][j] = parent->table[i][j];
            G = pNode->G + 1;
            int i = x;
            int j = y;
            table[i][j] = !table[i][j];
            if (s == 1) {
                table[i - 1][j] = !table[i - 1][j];
                table[i][j + 1] = !table[i][j + 1];
            } else if (s == 2) {
                table[i - 1][j] = !table[i - 1][j];
                table[i][j - 1] = !table[i][j - 1];
            } else if (s == 3) {
                table[i + 1][j] = !table[i + 1][j];
                table[i][j - 1] = !table[i][j - 1];
            } else {
                table[i + 1][j] = !table[i + 1][j];
                table[i][j + 1] = !table[i][j + 1];
            }
            int count_1_ = 0;
            for (int p = 0; p < N; p++)
                for (int q = 0; q < N; q++)
                    count_1_ += table[p][q];
            H = H_EXPRESSION;
            F = G + H;
        }
    }
    bool isTheSame(Node *n_ptr) {
        if (n_ptr->H != H)
            return false;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                if (table[i][j] != n_ptr->table[i][j])
                    return false;
            }
        return true;
    }
};

struct ptr_cmp {
    bool operator()(Node *&right, Node *&left) const {
        return right->F > left->F;
    }
};


class OpenSet {
public:
    priority_queue<Node *, vector<Node *>, ptr_cmp> open_set;

    Node *getMinFNodeAndDeleteIt() {
        Node *p = open_set.top();
        open_set.pop();
        return p;
    }
};

class CloseSet {
public:
    vector<Node *> close_set;
    bool isInCloseSet(Node *n_ptr) {
        for (auto i :close_set) {
            if (n_ptr->isTheSame((i)))
                return true;
        }
        return false;
    }
private:
};


bool processNodeAfterTrans(Node *node_ptr, OpenSet *open_ptr, CloseSet *close_ptr) {
    if (node_ptr->H == 0)
        return true;//ур╣╫жу╣Ц
    if (close_ptr->isInCloseSet(node_ptr)) {
        delete node_ptr;
        return false;
    }
    open_ptr->open_set.push(node_ptr);
    return false;
}

Node *nodeTransform(Node *n_ptr, OpenSet *open_ptr, CloseSet *close_ptr) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (i - 1 >= 0 && j - 1 >= 0) // type2 trans
            {
                Node *childNode_ptr = new Node(n_ptr, i, j, 2);
                if (processNodeAfterTrans(childNode_ptr, open_ptr, close_ptr))
                    return childNode_ptr;
            }
            if (i - 1 >= 0 && j + 1 < N) // type1 trans
            {
                Node *childNode_ptr = new Node(n_ptr, i, j, 1);
                if (processNodeAfterTrans(childNode_ptr, open_ptr, close_ptr))
                    return childNode_ptr;
            }
            if (i + 1 < N && j - 1 >= 0) // type3 trans
            {
                Node *childNode_ptr = new Node(n_ptr, i, j, 3);

                if (processNodeAfterTrans(childNode_ptr, open_ptr, close_ptr))
                    return childNode_ptr;
            }
            if (i + 1 < N && j + 1 < N) // type4 trans
            {
                Node *childNode_ptr = new Node(n_ptr, i, j, 4);

                if (processNodeAfterTrans(childNode_ptr, open_ptr, close_ptr))
                    return childNode_ptr;
            }
        }
    return nullptr;
}

int main() {
    for (int val = 0; val <= 9; val++) {
        const string input_file = "../input/input" + std::to_string(val) + ".txt";
        const string output_file = "../output/output" + std::to_string(val) + ".txt";
        ifstream f_in(input_file, ios::in);
        ofstream f_out(output_file);
        f_in >> N;
        Node *n_ptr = new Node(nullptr, -1, -1, 0);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                f_in >> n_ptr->table[i][j];

        n_ptr->G = 0;
        int count_1_ = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                count_1_ += n_ptr->table[i][j];

        n_ptr->H = H_EXPRESSION;
        n_ptr->F = n_ptr->H;

        OpenSet open_set;
        CloseSet close_set;

        open_set.open_set.push(n_ptr);

        while (!open_set.open_set.empty()) {
            Node *minFnode_ptr = open_set.getMinFNodeAndDeleteIt();
            close_set.close_set.push_back(minFnode_ptr);
            Node *res = nodeTransform(minFnode_ptr, &open_set, &close_set);
            if (res) {
                int steps = res->G;
                f_out << steps << endl;
                for (int i = 0; i < steps; i++) {

                    f_out << res->x << ',' << res->y << ',' << res->s << endl;
                    res = res->parent;
                }
                goto out;
            }
        }
        f_out << "No valid solution." << endl;
        out:;
    }
}
