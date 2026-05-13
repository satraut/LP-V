#include<iostream>
#include<stdlib.h>
#include<queue>
#include<omp.h>

using namespace std;

class node
{
public:
    int data;
    node *left, *right;
};

class Breadthfs
{
public:
    node* insert(node*, int);
    void bfs(node*);
};

node* Breadthfs::insert(node *root, int data)
{
    // If tree is empty
    if(root == NULL)
    {
        root = new node;
        root->data = data;
        root->left = NULL;
        root->right = NULL;

        return root;
    }

    queue<node*> q;
    q.push(root);

    while(!q.empty())
    {
        node *temp = q.front();
        q.pop();

        // Insert left child
        if(temp->left == NULL)
        {
            temp->left = new node;
            temp->left->data = data;
            temp->left->left = NULL;
            temp->left->right = NULL;

            return root;
        }
        else
        {
            q.push(temp->left);
        }

        // Insert right child
        if(temp->right == NULL)
        {
            temp->right = new node;
            temp->right->data = data;
            temp->right->left = NULL;
            temp->right->right = NULL;

            return root;
        }
        else
        {
            q.push(temp->right);
        }
    }

    return root;
}

void Breadthfs::bfs(node *head)
{
    queue<node*> q;

    q.push(head);

    while(!q.empty())
    {
        int qSize = q.size();

        // Parallel loop
        #pragma omp parallel for
        for(int i = 0; i < qSize; i++)
        {
            node *currNode;

            // Critical section for queue access
            #pragma omp critical
            {
                currNode = q.front();
                q.pop();

                cout << currNode->data << " ";
            }

            // Insert child nodes
            #pragma omp critical
            {
                if(currNode->left)
                {
                    q.push(currNode->left);
                }

                if(currNode->right)
                {
                    q.push(currNode->right);
                }
            }
        }
    }
}

int main()
{
    Breadthfs b;

    node *root = NULL;

    int data;
    char ans;

    do
    {
        cout << "\nEnter Data: ";
        cin >> data;

        root = b.insert(root, data);

        cout << "Do you want to insert more nodes? ";
        cin >> ans;

    }while(ans == 'y' || ans == 'Y');

    cout << "\nBFS Traversal is:\n";

    b.bfs(root);

    return 0;
}