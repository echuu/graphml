/*
 File description: 
 Author: Yabo Niu
 Last Edit: July 28, 2021
*/

#include <R.h>
#include <Rmath.h>

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <cmath>
#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <string.h>
// #include <limits.h>

#define max(a,b) ((a>b)?a:b)

void CheckPointer(void* pointer)
{
  if(pointer==NULL)
  {
    printf("Memory allocation error.\n");
    exit(1);
  }
  return;
}

//computes n choose m
int choose(int m, int n)
{
  if(m > n) return 0;
  if(m == 0) return 1;
  if(m == 1) return n;
  if(m == n) return 1;
  if(m == (n-1)) return n;
  
  return(choose(m, n-1) + choose(m-1, n-1));
}

//for qsort
int numeric (const void *p1, const void *p2)
{
  return(*((int*)p1) - *((int*)p2));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef class Graph* LPGraph;

class Graph
{
public:
  int nVertices; //number of vertices in the graph
  int** Edge;  //matrix containing the edges of the graph
  int* Labels; //identifies the connected components of the graph
  int nLabels; //number of labels or connected components
  int** Cliques; //storage for cliques
  int* CliquesDimens; //number of vertices in each clique
  int nCliques; //number of cliques
  
public:
  int** ConnectedComponents;
  int* ConnectedComponentsDimens;
  int nConnectedComponents; //number of connected components
public:
  int** StarComp;
  
public:  
  int* TreeEdgeA; //edges of the clique tree
  int* TreeEdgeB;
  int nTreeEdges; //number of edges in the generated clique tree
  
public:
  int   nMss; //the number of MSS that define the graph
  int** Mss; //storage for the MSS
  int*  MssDimens; //number of vertices in each MSS
  
public:  
  int* ordering;
  int** Separators; //storage for separators
  int* SeparatorsDimens;
  int nSeparators;
  
private:
  int* localord;
  
public:
  Graph(); //constructor
  Graph(LPGraph InitialGraph); //constructor 
  ~Graph(); //destructor 
  
public:  
  int  SearchVertex(); //identifies the next vertex to be eliminated
  
public:
  // int  ReadMss(char* sFileName); //read the MSS from file
  int  ReadMss(arma::umat EdgeMat); /* nyb */
void InitGraphFromMss(); //initialize the graph based on the MSS
void InitConnectedComponents();

public:  
  //the MSS (Minimal Sufficient Statistics) are the maximal cliques for our graph
  void InitGraph(int n);
  // int  ReadGraph(char* sFileName);
  // void WriteInfo(FILE* out);
  // void WriteInfo1(FILE* out);
  void GenerateCliques(int label);
  int  CheckCliques(); //checks whether each generated component
  //is complete in the given graph
  int  IsClique(int* vect,int nvect); //checks if the vertices in vect
  //form a clique in our graph
  // int  IsSubsetMss(int* vect,int nvect);
  void GenerateSeparators();
  void AttachLabel(int v, int label);
  void GenerateLabels();  
  int  GenerateAllCliques();
  int  IsDecomposable();
  void GetMPSubgraphs(); 
  //if the graph is decomposable, the mp-subgraphs will be the maximal cliques (the MSS)
  //otherwise, the minimum fill-in graph is generated
  //the mp-subgraphs will be stored in the Clique arrays
  // void FindCliqueTree();
  //this method should be called after calling GetMPSubgraphs(),
  //to init the clique tree if the graph is not decomposable
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef class SectionGraph* LPSectionGraph;

class SectionGraph : public Graph
{
public:
  int* Eliminated; //shows which vertices were eliminated from
  //the initial graph
  int nEliminated; //number of vertices we eliminated
  
  //methods
public:
  SectionGraph(LPGraph InitialGraph,int* velim); //constructor
  ~SectionGraph(); //destructor
  
public:
  int IsChain(int u,int v);//see if there is a chain between u and v
  //or, equivalently, checks if u and v are in the same connected component
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef class EliminationGraph* LPEliminationGraph;

class EliminationGraph : public Graph
{
public:
  int* Eliminated; //shows which vertices were eliminated from
  //the initial graph
  int nEliminated; //number of vertices we eliminated
  
  //methods
public:
  EliminationGraph(LPGraph InitialGraph,int vertex); //constructor
  ~EliminationGraph(); //destructor
public:	
  int  SearchVertex(); //identify a vertex to be eliminated		
public:
  void EliminateVertex(int x); //eliminates an extra vertex
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//constructs the minimum fill-in graph for a nondecomposable graph
LPGraph MakeFillInGraph(LPGraph graph)
{
  int u,v;
  int i;
  
  LPGraph gfill = new Graph(graph);
  CheckPointer(gfill);
  //if the graph is decomposable, there is no need to do anything
  if(gfill->IsDecomposable()) return gfill;
  
  int v1 = gfill->SearchVertex();
  //printf("v1 = %d\n",v1);
  //add edges to Def(Adj(x)) so that Adj(x) becomes a clique	
  for(u=0;u<gfill->nVertices;u++)
  {
    if(gfill->Edge[v1][u]==1)
    {
      for(v=u+1;v<gfill->nVertices;v++)
      {
        if((gfill->Edge[v1][v]==1)&&(gfill->Edge[u][v]==0))
        {
          gfill->Edge[v][u] = gfill->Edge[u][v] = 1;
          //printf("u = %d, v = %d\n",u,v);
        }	
      }	
    }	
  }		
  EliminationGraph egraph(graph,v1);
  for(i=1;i<graph->nVertices-1;i++)
  {
    v1 = egraph.SearchVertex();
    //printf("v1 = %d\n",v1);
    for(u=0;u<egraph.nVertices;u++)
    {
      if(egraph.Eliminated[u]) continue;
      if(egraph.Edge[v1][u]==1)
      {
        for(v=u+1;v<egraph.nVertices;v++)
        {
          if(egraph.Eliminated[v]) continue;
          if((egraph.Edge[v1][v]==1)&&(egraph.Edge[u][v]==0))
          {
            gfill->Edge[v][u] = gfill->Edge[u][v] = 1;
            //these are the edges that are added
            //to the initial graph
            //printf("u = %d, v = %d\n",u,v);
          }	
        }	
      }	
    }
    egraph.EliminateVertex(v1);
  }	
  return gfill;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//class Graph::Begins
Graph::Graph()
{
  nVertices                 = 0;
  Edge                      = NULL;
  Labels                    = NULL;
  nLabels                   = 0;
  Cliques                   = NULL;
  CliquesDimens             = NULL;
  nCliques                  = 0;
  StarComp                  = NULL;
  ConnectedComponents       = NULL;
  ConnectedComponentsDimens = NULL;
  nConnectedComponents      = 0;
  TreeEdgeA                 = NULL;
  TreeEdgeB                 = NULL;
  nTreeEdges                = 0;
  nMss                      = 0;
  Mss                       = NULL;
  MssDimens                 = NULL;
  ordering                  = NULL;
  Separators                = NULL;
  SeparatorsDimens          = NULL;
  nSeparators               = 0;
  localord                  = NULL;
  return;
}

Graph::Graph(LPGraph InitialGraph)
{
  nVertices                 = 0;
  Edge                      = NULL;
  Labels                    = NULL;
  nLabels                   = 0;
  Cliques                   = NULL;
  CliquesDimens             = NULL;
  nCliques                  = 0;
  ConnectedComponents       = NULL;
  ConnectedComponentsDimens = NULL;
  nConnectedComponents      = 0;
  StarComp                  = NULL;
  TreeEdgeA                 = NULL;
  TreeEdgeB                 = NULL;
  nTreeEdges                = 0;
  nMss                      = 0;
  Mss                       = NULL;
  MssDimens                 = NULL;
  ordering                  = NULL;
  Separators                = NULL;
  SeparatorsDimens          = NULL;
  nSeparators               = 0;
  localord                  = NULL;
  
  ///////////////////////////////////////
  int i,j;
  InitGraph(InitialGraph->nVertices);	
  for(i=0;i<nVertices;i++)
  {	
    for(j=0;j<nVertices;j++)
    {
      Edge[i][j] = InitialGraph->Edge[i][j];
    }
  }
  nMss      = InitialGraph->nMss;
  MssDimens = new int[nMss];
  CheckPointer(MssDimens);
  memset(MssDimens,0,nMss*sizeof(int));
  for(i=0;i<nMss;i++)
  {
    MssDimens[i] = InitialGraph->MssDimens[i];		
  }	
  Mss  = new int*[nMss];
  CheckPointer(Mss);
  memset(Mss,0,nMss*sizeof(int*));
  for(i=0;i<nMss;i++)
  {
    Mss[i] = new int[MssDimens[i]];
    CheckPointer(Mss[i]);
    memset(Mss[i],0,MssDimens[i]*sizeof(int));
    for(j=0;j<MssDimens[i];j++)
    {
      Mss[i][j] = InitialGraph->Mss[i][j];
    }	
  }	
  return;
}	

Graph::~Graph()
{
  int i;
  
  for(i=0; i<nVertices; i++)
  {
    delete[] Edge[i];
    Edge[i] = NULL; 
  }
  delete[] Edge; Edge = NULL;
  delete[] Labels; Labels = NULL;
  
  for(i=0; i<nVertices; i++)
  {
    delete[] Cliques[i];
    Cliques[i] = NULL;
  }
  delete[] Cliques; Cliques = NULL;
  delete[] CliquesDimens; CliquesDimens = NULL;
  
  if((nConnectedComponents>0)&&(ConnectedComponents!=NULL))
  {
    for(i=0;i<nConnectedComponents;i++)
    {  
      delete[] ConnectedComponents[i];
      ConnectedComponents[i] = NULL;
    }
    delete[] ConnectedComponents;
    ConnectedComponents = NULL;
    delete[] ConnectedComponentsDimens;
    ConnectedComponentsDimens = NULL;
  }
  if(NULL!=StarComp)
  {
    for(i=0;i<nTreeEdges;i++)
    {
      delete[] StarComp[i];
      StarComp[i] = NULL;
    }
    delete[] StarComp; StarComp = NULL;
  }
  delete[] TreeEdgeA; TreeEdgeA = NULL;
  delete[] TreeEdgeB; TreeEdgeB = NULL;	
  for(i=0;i<nMss;i++)
  {
    delete[] Mss[i];
    Mss[i] = NULL;
  }	
  delete[] Mss; Mss = NULL;
  delete[] MssDimens; MssDimens = NULL;	
  delete[] ordering; ordering = NULL;
  
  for(i=0; i<nVertices; i++)
  {
    delete[] Separators[i];
    Separators[i] = NULL;
  }
  delete[] Separators; Separators = NULL;
  delete[] SeparatorsDimens; SeparatorsDimens = NULL;
  delete[] localord; localord = NULL;
  return;
}

//REMEMBER!!!
//THIS FUNCTION DOES NOT INITIALIZE THE MSS COMPONENTS
//Should Call Function InitGraphFromMss() to properly initialize the graph
void Graph::InitGraph(int n)
{
  int i;
  
  nVertices = n;
  //alloc the matrix of vertices
  Edge = new int*[nVertices];
  CheckPointer(Edge);
  memset(Edge,0,nVertices*sizeof(int*));
  for(i=0; i<n; i++)
  {
    Edge[i] = new int[nVertices];
    CheckPointer(Edge[i]);
    memset(Edge[i],0,nVertices*sizeof(int));
  }
  nLabels = 0;
  Labels = new int[nVertices];
  CheckPointer(Labels);
  memset(Labels,0,nVertices*sizeof(int));
  nCliques = 0;
  Cliques = new int*[nVertices];
  CheckPointer(Cliques);
  memset(Cliques,0,nVertices*sizeof(int*)); 
  for(i=0;i<n;i++)
  {
    Cliques[i] = new int[nVertices];
    CheckPointer(Cliques[i]);
    memset(Cliques[i],0,nVertices*sizeof(int));
  }	
  CliquesDimens = new int[nVertices];
  CheckPointer(CliquesDimens);
  memset(CliquesDimens,0,nVertices*sizeof(int));
  nTreeEdges = 0;
  TreeEdgeA = new int[nVertices];
  CheckPointer(TreeEdgeA);
  memset(TreeEdgeA,0,nVertices*sizeof(int));
  TreeEdgeB = new int[nVertices];
  CheckPointer(TreeEdgeB);
  memset(TreeEdgeB,0,nVertices*sizeof(int));
  ordering = new int[nVertices];
  CheckPointer(ordering);
  memset(ordering,0,nVertices*sizeof(int));
  Separators = new int*[nVertices];
  CheckPointer(Separators);
  memset(Separators,0,nVertices*sizeof(int*));
  for(i=0; i<n; i++)
  {
    Separators[i] = new int[nVertices];
    CheckPointer(Separators[i]);
    memset(Separators[i],0,nVertices*sizeof(int));
  }
  SeparatorsDimens = new int[nVertices];
  CheckPointer(SeparatorsDimens);
  memset(SeparatorsDimens,0,nVertices*sizeof(int));
  localord = new int[nVertices];
  CheckPointer(localord);
  memset(localord,0,nVertices*sizeof(int));
  return;
}

/* nyb */
int Graph::ReadMss(arma::umat EdgeMat)
{
  int i, j;
  
  nMss = EdgeMat.n_rows;
  MssDimens = new int[nMss];
  CheckPointer(MssDimens);	
  memset(MssDimens,0,nMss*sizeof(int));	
  Mss = new int*[nMss];
  CheckPointer(Mss);
  memset(Mss,0,nMss*sizeof(int*));
  
  for(i=0; i<nMss; i++)
  {
    MssDimens[i] = EdgeMat(i,0);
    Mss[i] = new int[MssDimens[i]];
    CheckPointer(Mss[i]);
    memset(Mss[i],0,MssDimens[i]*sizeof(int));
    for(j=0; j<MssDimens[i]; j++)
    {
      Mss[i][j] = EdgeMat(i,j+1)-1; // index begins at zero
    }		
    // the mss need to be sorted
    qsort((void*)Mss[i],MssDimens[i], sizeof(int), numeric);		
  }	
  return 1;
}	

void Graph::InitGraphFromMss()
{
  int i,j,k;
  int n = 0;
  //first determine the number of vertices
  //remember that the Mss are now sorted in ascending order
  for(i=0;i<nMss;i++)
  {
    n = max(n,Mss[i][MssDimens[i]-1]);
  }
  InitGraph(n+1); //since the index started at 0
  //now determine the adjacency matrix based on the Mss
  for(i=0;i<nMss;i++)
  {
    for(j=0;j<MssDimens[i];j++)
    {
      for(k=j+1;k<MssDimens[i];k++)
      {
        Edge[Mss[i][j]][Mss[i][k]] = 1;
        Edge[Mss[i][k]][Mss[i][j]] = 1;
      }	
    }	
  }
  return;
}	

void Graph::GenerateCliques(int label)
{
  int i, j, k, p, r;
  int n = nVertices;
  int* clique = new int[nVertices];
  CheckPointer(clique);
  int* LRound = new int[nVertices];
  CheckPointer(LRound);
  
  //clean memory
  memset(localord,0,nVertices*sizeof(int));
  memset(clique,0,nVertices*sizeof(int));
  memset(LRound,0,nVertices*sizeof(int));
  for(i=0;i<n;i++)
  {
    memset(Cliques[i],0,nVertices*sizeof(int));
  }
  memset(CliquesDimens,0,nVertices*sizeof(int));
  
  int v, vk;
  int PrevCard = 0;
  int NewCard;
  int s = -1;
  
  for(i = n-1; i>=0; i--)
  {
    NewCard = -1;
    //choose a vertex v...
    for(j=0; j<n; j++)
    {
      //test vertex j
      if((Labels[j] == label) && (LRound[j] == 0))
      {
        int maxj = 0;
        for(r=0; r<n; r++)
        {
          if(Labels[r] == label)
          {
            if(Edge[j][r] && LRound[r])
            {
              maxj++;
            }
          }
        }
        
        if(maxj > NewCard)
        {
          v = j;
          NewCard = maxj;
        }
      }
    }
    
    if(NewCard == -1)
    {
      break;
    }
    
    localord[v] = i;
    if(NewCard <= PrevCard)
    {
      //begin new clique
      s++;
      for(r=0; r<n; r++)
      {
        if(Labels[r] == label)
        {
          if(Edge[v][r] && LRound[r])
          {
            Cliques[s][CliquesDimens[s]] = r;
            CliquesDimens[s]++;
          }
        }
      }
      if(NewCard != 0)
      {
        //get edge to parent
        vk = Cliques[s][0];
        k  = localord[vk];
        for(r=1; r<CliquesDimens[s]; r++)
        {
          if(localord[Cliques[s][r]] < k)
          {
            vk = Cliques[s][r];
            k  = localord[vk];
          }
        }				
        p = clique[vk];
        TreeEdgeA[nTreeEdges] = s;
        TreeEdgeB[nTreeEdges] = p;
        nTreeEdges++;
      }
    }		
    clique[v] = s;
    Cliques[s][CliquesDimens[s]] = v;
    CliquesDimens[s]++;
    LRound[v] = 1;
    PrevCard = NewCard;
  }
  
  nCliques = s+1;
  
  delete[] clique;
  delete[] LRound;
  return;
}

int Graph::CheckCliques()
{
  int i, j, k;
  
  for(i=0; i<nCliques; i++)
  {
    for(j = 0; j<CliquesDimens[i]-1; j++)
    {
      for(k = j+1; k<CliquesDimens[i]; k++)
      {
        if(Edge[Cliques[i][j]][Cliques[i][k]] == 0)
        {
          return(-i-1);
        }
      }
    }
    qsort((void*)Cliques[i], CliquesDimens[i], sizeof(int), numeric);
  }
  
  return 1;
}

int Graph::IsClique(int* vect,int nvect)
{
  int i,j;
  int okay = 1;
  for(i=0;i<nvect;i++)
  {
    for(j=i+1;j<nvect;j++)
    {
      if(Edge[vect[i]][vect[j]]==0)
      {
        okay = 0; break;				
      }
    }
    if(!okay) break;
  }
  return okay;
}

void Graph::GenerateSeparators()
{
  int i;
  int j, k;
  int FirstClique, SecondClique;
  int v;
  
  for(i=0; i<nTreeEdges; i++)
  {
    FirstClique = TreeEdgeA[i];
    SecondClique = TreeEdgeB[i];
    
    for(j=0; j<CliquesDimens[FirstClique]; j++)
    {
      v = Cliques[FirstClique][j];
      for(k=0; k<CliquesDimens[SecondClique]; k++)
      {
        if(v == Cliques[SecondClique][k])
        {
          Separators[i][SeparatorsDimens[i]] = v;
          SeparatorsDimens[i]++;
          break;
        }
      }
    }
    qsort((void*)Separators[i], SeparatorsDimens[i], sizeof(int), numeric);
  }	
  return;
}

void Graph::AttachLabel(int v, int label)
{
  int i;
  
  //only if v has not been labeled yet
  if(Labels[v] == 0)
  {
    Labels[v] = label;
    for(i=0; i<nVertices; i++)
    {
      if(Edge[v][i] == 1)
      {
        AttachLabel(i, label);
      }
    }
  }	
  return;
}

void Graph::GenerateLabels()
{
  int i;
  int NotFinished = 1;
  int label = 0;
  int v;
  
  memset(Labels,0,nVertices*sizeof(int));
  nLabels = 0;
  while(NotFinished)
  {
    v = -1;
    for(i=0; i<nVertices; i++)
    {
      if(Labels[i] == 0)
      {
        v = i;
        break;
      }
    }
    
    if(v == -1)
    {
      NotFinished = 0;
    }
    else
    {
      label++;
      AttachLabel(v, label);
    }
  }	
  nLabels = label;
  return;
}

int Graph::GenerateAllCliques()
{
  int i, j;
  int n = nVertices;
  int label;
  int nAssigned = 0;
  
  //Alloc Memory :: Begin
  int nAllCliques  = 0;
  int** AllCliques = new int*[n];
  CheckPointer(AllCliques);
  memset(AllCliques,0,n*sizeof(int*));
  for(i=0;i<n;i++)
  {
    AllCliques[i] = new int[n];
    CheckPointer(AllCliques[i]);
    memset(AllCliques[i],0,n*sizeof(int));
  }
  
  int* AllCliquesDimens = new int[n];
  CheckPointer(AllCliquesDimens);
  memset(AllCliquesDimens,0,n*sizeof(int));
  
  int nAllTreeEdges = 0;
  int* AllTreeEdgeA = new int[n];
  CheckPointer(AllTreeEdgeA);
  memset(AllTreeEdgeA,0,n*sizeof(int));
  int* AllTreeEdgeB = new int[n];
  CheckPointer(AllTreeEdgeB);
  memset(AllTreeEdgeB,0,n*sizeof(int));
  
  int** AllSeparators = new int*[n];
  CheckPointer(AllSeparators);
  memset(AllSeparators,0,n*sizeof(int*));
  for(i=0;i<n;i++)
  {
    AllSeparators[i] = new int[n];
    CheckPointer(AllSeparators[i]);
    memset(AllSeparators[i],0,n*sizeof(int));
  }
  int nAllSeparators = 0;	
  int* AllSeparatorsDimens = new int[n];
  CheckPointer(AllSeparatorsDimens);
  memset(AllSeparatorsDimens,0,n*sizeof(int));
  //Alloc Memory :: End
  
  //clean memory
  nCliques = 0;	
  for(i=0;i<n;i++)
  {		
    memset(Cliques[i],0,n*sizeof(int));
  }		
  memset(CliquesDimens,0,n*sizeof(int));
  nTreeEdges = 0;	
  memset(TreeEdgeA,0,n*sizeof(int));	
  memset(TreeEdgeB,0,n*sizeof(int));	
  memset(ordering,0,n*sizeof(int));	
  for(i=0; i<n; i++)
  {
    memset(Separators[i],0,n*sizeof(int));
  }	
  memset(SeparatorsDimens,0,n*sizeof(int));	
  
  //find all the connected components	
  GenerateLabels();
  
  for(label = 1; label<=nLabels; label++)
  {		
    GenerateCliques(label);
    if(CheckCliques() < 0)
    {
      for(i=0; i<n; i++)
      {
        delete[] AllCliques[i];
        AllCliques[i] = NULL;
      }
      delete[] AllCliques; AllCliques = NULL;
      delete[] AllCliquesDimens; AllCliquesDimens = NULL;
      delete[] AllTreeEdgeA; AllTreeEdgeA = NULL;
      delete[] AllTreeEdgeB; AllTreeEdgeB = NULL;
      for(i=0; i<n; i++)
      {
        delete[] AllSeparators[i];
        AllSeparators[i] = NULL;
      }
      delete[] AllSeparators; AllSeparators = NULL;
      delete[] AllSeparatorsDimens; AllSeparatorsDimens = NULL;
      
      return 0; //this is not a decomposable model
    }
    GenerateSeparators();		
    //store the newly generated cliques
    for(i=0; i<nTreeEdges; i++)
    {
      AllTreeEdgeA[nAllTreeEdges] = nAllCliques + TreeEdgeA[i];
      AllTreeEdgeB[nAllTreeEdges] = nAllCliques + TreeEdgeB[i];
      TreeEdgeA[i] = 0;
      TreeEdgeB[i] = 0;
      nAllTreeEdges++;
    }		
    for(i=0; i<nCliques; i++)
    {
      for(j=0; j<CliquesDimens[i]; j++)
      {
        AllCliques[nAllCliques][j] = Cliques[i][j];
        Cliques[i][j] = 0;
      }
      AllCliquesDimens[nAllCliques] = CliquesDimens[i];
      CliquesDimens[i] = 0;
      nAllCliques++;
    }
    nCliques = 0;
    
    for(i=0; i<nTreeEdges; i++)
    {
      for(j=0; j<SeparatorsDimens[i]; j++)
      {
        AllSeparators[nAllSeparators][j] = Separators[i][j];
        Separators[i][j] = 0;
      }
      AllSeparatorsDimens[nAllSeparators] = SeparatorsDimens[i];
      SeparatorsDimens[i] = 0;
      nAllSeparators++;
    }
    /*
     //add an extra (null) separator between two connected components
     if(label<nLabels)
     {	
     AllSeparatorsDimens[nAllSeparators] = 0;
     nAllSeparators++;
     }
     */	
    //clean memory
    nSeparators = 0;		
    nTreeEdges = 0;
    
    //printf("Perfect ordering :: ");
    int partialAssigned = 0;
    for(i=0; i<n; i++)
    {
      //printf("%d ",localord[i]);
      if(Labels[i] == label)
      {	
        ordering[i] = localord[i] - nAssigned;
        partialAssigned++;
      }
    }
    //printf("\n");
    nAssigned += partialAssigned;
  }	
  for(i=0; i<nAllCliques; i++)
  {
    for(j=0; j<AllCliquesDimens[i]; j++)
    {
      Cliques[nCliques][j] = AllCliques[i][j];
    }
    CliquesDimens[nCliques] = AllCliquesDimens[i];
    nCliques++;
  }
  
  for(i=0; i<nAllTreeEdges; i++)
  {
    TreeEdgeA[nTreeEdges] = AllTreeEdgeA[i];
    TreeEdgeB[nTreeEdges] = AllTreeEdgeB[i];
    nTreeEdges++;
  }
  
  for(i=0; i<nAllSeparators; i++)
  {
    for(j=0; j<AllSeparatorsDimens[i]; j++)
    {
      Separators[nSeparators][j] = AllSeparators[i][j];
    }
    SeparatorsDimens[nSeparators] = AllSeparatorsDimens[i];
    nSeparators++;
  }
  //free memory
  for(i=0; i<n; i++)
  {
    delete[] AllCliques[i];
    AllCliques[i] = NULL;
  }
  delete[] AllCliques; AllCliques = NULL;
  delete[] AllCliquesDimens; AllCliquesDimens = NULL;
  delete[] AllTreeEdgeA; AllTreeEdgeA = NULL;
  delete[] AllTreeEdgeB; AllTreeEdgeB = NULL;
  for(i=0; i<n; i++)
  {
    delete[] AllSeparators[i];
    AllSeparators[i] = NULL;
  }
  delete[] AllSeparators; AllSeparators = NULL;
  delete[] AllSeparatorsDimens; AllSeparatorsDimens = NULL;
  return 1;
}

int Graph::SearchVertex()
{
  int x, u, v;
  int okay;
  int* sxAdj = new int[nVertices];
  CheckPointer(sxAdj);
  memset(sxAdj,0,nVertices*sizeof(int));
  
  for(x=0;x<nVertices;x++)
  {
    memmove(sxAdj,Edge[x],nVertices*sizeof(int));	
    sxAdj[x] = 1;
    okay = 1;
    for(u=0;u<nVertices;u++)
    {
      if((u!=x)&&(Edge[x][u]==1))
      {
        sxAdj[u] = 0; //we take u out
        for(v=u+1;v<nVertices;v++)
        {
          if((v!=x)&&(Edge[x][v]==1)&&(Edge[u][v]==0))
          {
            sxAdj[v] = 0;//we take v out
            SectionGraph sgraph(this,sxAdj);
            okay = sgraph.IsChain(u,v);
            sxAdj[v] = 1;//now put v back in the adjacency list of x
          }
          if(!okay) break;
        }
        sxAdj[u] = 1; //we put u back
      }
      if(!okay) break;
    }
    if(okay) break;
  }
  delete[] sxAdj;
  if(x==nVertices) x = -1;
  return x;
}

int Graph::IsDecomposable()
{
  return GenerateAllCliques();
}

void Graph::InitConnectedComponents()
{
  int i,label;
  if(ConnectedComponents !=NULL){
    for(i=0; i< nConnectedComponents; i++){
      if(ConnectedComponents[i]!=NULL) delete[] ConnectedComponents[i];}
    delete[] ConnectedComponents;}
  if(ConnectedComponentsDimens!=NULL) delete[] ConnectedComponentsDimens;
  
  nConnectedComponents=nLabels;
  
  ConnectedComponents = new int*[nConnectedComponents];
  CheckPointer(ConnectedComponents);
  
  ConnectedComponentsDimens = new int[nConnectedComponents];
  CheckPointer(ConnectedComponentsDimens);
  for(label=1;label<=nLabels;label++)
  {
    //count the number of vertices being labeled with label
    int count=0;
    for(i=0;i<nVertices;i++)
    {
      if(Labels[i]==label) count++;  
    }
    //printf("label = %d :: count = %d\n",label,count);
    ConnectedComponentsDimens[label-1]=count;
    ConnectedComponents[label-1] = new int[count];
    CheckPointer(ConnectedComponents[label-1]);
    count=0;
    for(i=0;i<nVertices;i++)
    {
      if(Labels[i]==label)
      {
        ConnectedComponents[label-1][count]=i;
        count++;
      }  
    }         
  }
  return;
}

void Graph::GetMPSubgraphs()
{
  int i,j;
  
  if(IsDecomposable()) return;//easy task if the graph is decomposable
  //if not, generate the minimal fill-in graph
  LPGraph gfill = MakeFillInGraph(this);
  if(!gfill->IsDecomposable())
  {
    printf("The fill-in graph is not decomposable!\n Something is wrong.\n");
    exit(1);
  }	
  //gfill->WriteInfo(stdout);
  //we clean the memory a bit, just to be on the safe side
  nCliques = nSeparators = 0;	
  for(i=0;i<nVertices;i++)
  {		
    memset(Cliques[i],0,nVertices*sizeof(int));
    memset(Separators[i],0,nVertices*sizeof(int));
  }		
  memset(CliquesDimens,0,nVertices*sizeof(int));
  memset(SeparatorsDimens,0,nVertices*sizeof(int));
  nTreeEdges = 0;	
  memset(TreeEdgeA,0,nVertices*sizeof(int));	
  memset(TreeEdgeB,0,nVertices*sizeof(int));	
  memset(ordering,0,nVertices*sizeof(int));					
  //////////////////////////////////////////////////////////
  //done cleaning memory                                  //
  //////////////////////////////////////////////////////////	
  int* UsedEdge = new int[gfill->nTreeEdges]; CheckPointer(UsedEdge);
  //mark the edges as "not used"
  memset(UsedEdge,0,gfill->nTreeEdges*sizeof(int));   
  int* MarkC = new int[gfill->nCliques]; CheckPointer(MarkC); 
  memset(MarkC,0,gfill->nCliques*sizeof(int));
  int* MarkS = new int[gfill->nSeparators]; CheckPointer(MarkS);
  memset(MarkS,0,gfill->nSeparators*sizeof(int)); 
  
  //printf("nTreeEdges = %d\n",gfill->nTreeEdges);
  ////////////////////////////////////////////////////////////   
  while(1)
  {
    //identify a terminal clique Cj
    int edg;
    int Ci;
    int Cj;
    for(edg=0;edg<gfill->nTreeEdges;edg++)
    {
      //if we already used that edge, go to the next one
      if(UsedEdge[edg]) continue;
      Ci = gfill->TreeEdgeB[edg];
      Cj = gfill->TreeEdgeA[edg];
      //printf("Cj = %d\n",Cj+1);         
      int foundterminal = 1;
      for(i=0;i<gfill->nTreeEdges;i++)
      {
        if(UsedEdge[i]) continue;
        if(gfill->TreeEdgeB[i]==Cj)
        {
          foundterminal=0;
          break;
        } 
      }
      if(foundterminal) break;
    }
    if(edg==gfill->nTreeEdges) break;
    //printf("Cj = %d :: Ci = %d\n",Cj+1,Ci+1);
    //mark the edge as used
    UsedEdge[edg]=1;
    //Step 4
    if(IsClique(gfill->Separators[edg],
                gfill->SeparatorsDimens[edg]))
    {
      //printf("%d is clique\n",edg+1);
      MarkC[Cj]  = 1;
      MarkS[edg] = 1;  
    }
    else
    {
      MarkC[Cj] = -1;
      //combine the delta sets associated with Ci and Cj
      int  len1 = gfill->CliquesDimens[Ci]+
        gfill->CliquesDimens[Cj];
      int  len2 = 0;
      int* buffer1 = new int[len1]; CheckPointer(buffer1);		
      int* buffer2 = new int[len1]; CheckPointer(buffer2);
      len1 = 0;
      for(i=0;i<gfill->CliquesDimens[Ci];i++)
      {
        buffer1[len1] = gfill->Cliques[Ci][i];
        len1++;   
      }
      for(i=0;i<gfill->CliquesDimens[Cj];i++)
      {
        buffer1[len1] = gfill->Cliques[Cj][i];
        len1++;
      }
      qsort((void*)buffer1,len1,sizeof(int),numeric);
      buffer2[len2]=buffer1[0];
      for(i=0;i<len1;i++)
      {
        if(buffer2[len2]<buffer1[i])
        {
          len2++;
          buffer2[len2]=buffer1[i];
        }	
      }
      len2++;
      for(i=0;i<len2;i++)
      {
        gfill->Cliques[Ci][i] = buffer2[i];
      } 
      gfill->CliquesDimens[Ci] = len2;    
      ///////////////
      delete[] buffer1;
      delete[] buffer2;
    }
  }
  for(i=0;i<gfill->nCliques;i++)
  {
    if(MarkC[i]==-1) continue;
    for(j=0;j<gfill->CliquesDimens[i];j++)
    {
      Cliques[nCliques][j] = gfill->Cliques[i][j];
    }
    CliquesDimens[nCliques] = gfill->CliquesDimens[i];
    nCliques++;
  }
  for(i=0;i<gfill->nSeparators;i++)
  {
    if(MarkS[i]==0) continue;
    for(j=0;j<gfill->SeparatorsDimens[i];j++)
    {
      Separators[nSeparators][j] = gfill->Separators[i][j];
    }
    SeparatorsDimens[nSeparators] = gfill->SeparatorsDimens[i];
    nSeparators++;
  }
  //////////////////////////////////////////////////
  delete[] MarkS;
  delete[] MarkC;
  delete[] UsedEdge;
  delete gfill;	
  return;
}
//class Graph::Ends

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//class SectionGraph::Begins
SectionGraph::SectionGraph(LPGraph InitialGraph,int* velim) : Graph(InitialGraph)
{
  int i,j;
  
  Eliminated = new int[nVertices];
  CheckPointer(Eliminated);
  memset(Eliminated,0,nVertices*sizeof(int));
  nEliminated = 0;
  for(i=0;i<nVertices;i++)
  {
    if(velim[i])
    {	
      Eliminated[i] = 1;
      nEliminated++;
    }	
  }
  //delete all the edges corresponding to the vertices
  //we eliminated
  for(i=0;i<nVertices;i++)
  {		
    if(Eliminated[i])
    {
      for(j=0;j<nVertices;j++)
      {
        if(1==Edge[i][j])
        {
          Edge[i][j] = Edge[j][i] = 0;
        }	
      }	
    }	
  }
  return;
}

SectionGraph::~SectionGraph()
{
  delete[] Eliminated;
  nEliminated = 0;
  return;
}	

int SectionGraph::IsChain(int u,int v)
{
  if(nLabels==0)
  {	
    GenerateLabels();
  }	
  if(Eliminated[u] || Eliminated[v])
  {
    printf("One of the vertices %d,%d has been eliminated...\n",u,v);
    exit(1);
  }		
  if(Labels[u]==Labels[v]) return 1;
  return 0;
}	
//class SectionGraph::Ends

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//class EliminationGraph::Begins
EliminationGraph::EliminationGraph(LPGraph InitialGraph,int vertex) : Graph(InitialGraph)
{	
  Eliminated = new int[nVertices];
  CheckPointer(Eliminated);
  memset(Eliminated,0,nVertices*sizeof(int));
  nEliminated = 0;
  EliminateVertex(vertex);
  return;
}

EliminationGraph::~EliminationGraph()
{
  delete[] Eliminated;
  nEliminated = 0;
  return;
}

void EliminationGraph::EliminateVertex(int x)
{
  int i,j;
  
  //adding edges in Def(Adj(x)) so that Adj(x) becomes a clique
  for(i=0;i<nVertices;i++)
  {
    if((i!=x)&&(!Eliminated[i])&&(Edge[x][i]==1))
    {
      for(j=i+1;j<nVertices;j++)
      {
        if((j!=x)&&(!Eliminated[j])&&(Edge[x][j]==1)&&(Edge[i][j]==0))
        {
          Edge[i][j] = Edge[j][i] = 1;
        }	
      }	
    }	
  }	
  
  //eliminate all edges incident to x
  for(i=0;i<nVertices;i++)
  {
    if((i!=x)&&(!Eliminated[i])&&(Edge[x][i]==1))
    {
      Edge[x][i] = Edge[i][x] = 0;
    }	
  }	
  
  //eliminate vertex x
  Eliminated[x] = 1;
  nEliminated++;
  return;
}

int EliminationGraph::SearchVertex()
{
  int x, u, v;
  int okay;
  int* sxAdj = new int[nVertices];
  CheckPointer(sxAdj);
  memset(sxAdj,0,nVertices*sizeof(int));
  
  for(x=0;x<nVertices;x++)
  {
    if(Eliminated[x]) continue;
    memmove(sxAdj,Edge[x],nVertices*sizeof(int));	
    sxAdj[x] = 1;
    okay = 1;
    for(u=0;u<nVertices;u++)
    {
      if(Eliminated[u]) continue;
      if((u!=x)&&(Edge[x][u]==1))
      {
        sxAdj[u] = 0; //we take u out
        for(v=u+1;v<nVertices;v++)
        {
          if(Eliminated[v]) continue;
          if((v!=x)&&(Edge[x][v]==1)&&(Edge[u][v]==0))
          {
            sxAdj[v] = 0;//we take v out
            SectionGraph sgraph(this,sxAdj);
            okay = sgraph.IsChain(u,v);
            sxAdj[v] = 1;//now put v back in the adjacency list of x
          }
          if(!okay) break;
        }
        sxAdj[u] = 1; //we put u back
      }
      if(!okay) break;
    }
    if(okay) break;
  }
  delete[] sxAdj;
  if(x==nVertices) x = -1;
  return x;
}
//class EliminationGraph::Ends

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
Rcpp::List getJT(arma::umat EdgeMat)
{
  LPGraph graph = new Graph;
  graph->ReadMss(EdgeMat);
  graph->InitGraphFromMss();
  graph->GetMPSubgraphs();
  graph->InitConnectedComponents();
  
  vector< arma::uvec > Clist;
  vector< arma::uvec > Slist;
  Clist.clear();
  Slist.clear();
  Clist.shrink_to_fit();
  Slist.shrink_to_fit();
  
  for(unsigned int i=0; i<graph->nCliques; i++){
    arma::uvec temp(graph->CliquesDimens[i], arma::fill::zeros);
    for(unsigned int j=0; j<graph->CliquesDimens[i]; j++){
      temp(j) = graph->Cliques[i][j] + 1;
    }
    Clist.push_back(temp);
  }
  
  // arma::uvec a(1); a(0) = 99999; Slist.push_back(a);
  for(unsigned int i=0; i<graph->nSeparators; i++){
    arma::uvec temp(graph->SeparatorsDimens[i], arma::fill::zeros);
    for(unsigned int j=0; j<graph->SeparatorsDimens[i]; j++){
      temp(j) = graph->Separators[i][j] + 1;
    }
    Slist.push_back(temp);
  }
  /*
   cout<<"printing adjacency matrix..."<<endl;
   for(int i=0; i<graph->nVertices; i++)
   {
   for(int j=0; j<graph->nVertices; j++)
   {
   printf("%d  ", graph->Edge[i][j]);
   }
   printf("\n");
   }
   printf("\n"); */
  cout<<"Number of Cliques: "<<graph->nCliques<<endl;
  cout<<"Number of Separators: "<<graph->nSeparators<<endl;
  
  delete graph;
  return Rcpp::List::create(Rcpp::Named("Cliques") = Clist,
                            Rcpp::Named("Separators") = Slist);
}

// internal version of the function getJT()
void getJT_C(arma::umat EdgeMat, 
             vector<arma::uvec> &Clist, vector<arma::uvec> &Slist, int &nC, int &nS)
{
  LPGraph graph = new Graph;
  graph->ReadMss(EdgeMat);
  graph->InitGraphFromMss();
  graph->GetMPSubgraphs();
  graph->InitConnectedComponents();
  
  Clist.clear();
  Slist.clear();
  Clist.shrink_to_fit();
  Slist.shrink_to_fit();
  
  for(unsigned int i=0; i<graph->nCliques; i++){
    arma::uvec temp(graph->CliquesDimens[i], arma::fill::zeros);
    for(unsigned int j=0; j<graph->CliquesDimens[i]; j++){
      temp(j) = graph->Cliques[i][j] + 1;
    }
    Clist.push_back(temp);
  }
  
  for(unsigned int i=0; i<graph->nSeparators; i++){
    arma::uvec temp(graph->SeparatorsDimens[i], arma::fill::zeros);
    for(unsigned int j=0; j<graph->SeparatorsDimens[i]; j++){
      temp(j) = graph->Separators[i][j] + 1;
    }
    Slist.push_back(temp);
  }
  
  nC = graph->nCliques;
  nS = graph->nSeparators;
  
  delete graph;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/* the core function of calculating the normalizing constant of a G-Wishart density */
// [[Rcpp::export]]
arma::vec log_exp_mc(arma::umat G, arma::uvec nu, unsigned int b, arma::mat H, 
                     unsigned int check_H, unsigned int mc, unsigned int p){
  unsigned int iter, i, j, h, r;
  unsigned int mc_iter = mc, dim = p, b_c = b;
  double sumPsi, sumPsiH, sumPsiHi, sumPsiHj;
  arma::mat psi(p, p, arma::fill::zeros);
  arma::vec f_T(mc_iter, arma::fill::zeros);
  
  if(check_H==1)
  {
    
    for(iter=0; iter<mc_iter; iter++){
      for(i=0; i<dim; i++) { psi(i,i) = sqrt(R::rgamma((b_c+nu(i))/2.0, 2.0)); }
      
      for(i=0; i<(dim-1); i++){
        for(j=i+1; j<dim; j++){
          if(G(i,j)==1) { psi(i,j) = R::rnorm(0,1); }
          else { psi(i,j) = 0.0; }
        }
      }
      
      for(i=0; i<(dim-1); i++){
        for(j=i+1; j<dim; j++){
          if(G(i,j)==0){
            psi(i,j) = 0.0;
            if(i>0){
              sumPsi = 0.0;
              for(h=0; h<(i-1); h++){ sumPsi = sumPsi + psi(h,i)*psi(h,j); }
              psi(i,j) = -sumPsi/psi(i,i);
            }
            f_T(iter) = f_T(iter) + psi(i,j)*psi(i,j);
          }
        }
      }
    }
    
  }
  else
  {
    
    for(iter=0; iter<mc_iter; iter++){
      for(i=0; i<dim; i++){ psi(i,i) = sqrt(R::rgamma((b_c+ nu(i))/2.0, 2.0)); }
      
      for(i=0; i<(dim-1); i++){
        for(j=i+1; j<dim; j++){
          if(G(i,j)==1) { psi(i,j) = R::rnorm(0, 1); } 
          else { psi(i,j) = 0.0; }
        }
      }
      
      for(i=0; i<(dim-1); i++){
        for(j=i+1; j<dim; j++){
          if(G(i,j)==0){
            sumPsiH = 0.0;
            for(h=i; h<j; h++){ sumPsiH = sumPsiH + psi(i,h)*H(h,j); }
            psi(i,j) = -sumPsiH;
            
            if(i>0){
              for(r=0; r<i; r++){
                sumPsiHi = 0.0;
                for(h=r; h<(i+1); h++){ sumPsiHi = sumPsiHi + psi(r,h)*H(h,i); }
                sumPsiHj = 0.0;
                for(h=r; h<(j+1); h++){ sumPsiHj = sumPsiHj + psi(r,h)*H(h,j); }
                psi(i,j) = psi(i,j) - (sumPsiHi*sumPsiHj)/psi(i,i);
              }
            }
            f_T(iter) = f_T(iter) + psi(i,j)*psi(i,j);
          }
        }
      }
    }
    
  }
  
  return f_T;
}

/* get the normalizing constant of a G-Wishart density */
// [[Rcpp::export]]
double gnorm_c(arma::umat Adj, double b, arma::mat D, unsigned int iter){
  unsigned int p = Adj.n_rows;
  arma::umat A0 = Adj;
  // arma::umat Ip = arma::eye<arma::umat>(p,p);
  // A0 = A0 - Ip;
  arma::umat G = arma::trimatu(A0);
  
  arma::mat Ti = arma::chol(inv(D));
  arma::mat T(p, p, arma::fill::zeros);
  arma::vec Ones(p, arma::fill::ones);
  for(unsigned int i=0; i<p; i++){ T.col(i) = Ti(i,i)*Ones; }
  arma::mat H = Ti / T;
  unsigned int check_H;
  arma::mat I = arma::eye<arma::mat>(p,p);
  if(arma::accu(abs(H-I))==0) { check_H = 1; }
  else { check_H = 0; }
  
  arma::uvec nu = arma::sum(G, 1);
  unsigned int size_graph = arma::accu(G);
  double logIg;
  
  // For the case, G is a full graph
  if(size_graph==p*(p-1)/2){
    arma::vec nu1 = arma::conv_to<arma::vec>::from(nu);
    logIg = 0.5*size_graph*log(M_PI) + 0.5*p*(b+p-1)*log(2) + sum(arma::lgamma(0.5*(b+nu1))) - 0.5*(b+p-1)*log(det(D));
    // [wrong] logIg = 0.5*size_graph*log(M_PI) + 0.5*p*(b+p-1)*log(2) + sum(arma::lgamma(0.5*(b+nu1))) + 0.5*(b+p-1)*log(det(D));
    /* nyb */
  }
  
  // For the case, G is an empty graph
  if(size_graph==0){
    logIg = 0.5*p*b*log(2) + p*R::lgammafn(0.5*b) - 0.5*b*arma::sum(log(D.diag()));
  }
  
  // For the case G is NOT full graph
  if( (size_graph!=p*(p-1)/2) && (size_graph!=0) ){
    arma::vec f_T = log_exp_mc(G, nu, b, H, check_H, iter, p);
    double c = -0.5*min(f_T)-log(iter);
    double log_Ef_T = c + log(sum( exp( -log(iter)-0.5*f_T-c ) ));
    // double log_Ef_T = log(mean(exp(-0.5*f_T)));
    arma::vec nu2 = arma::conv_to<arma::vec>::from(nu);
    arma::urowvec temp1 = sum(G, 0);
    arma::vec temp2 = arma::conv_to<arma::vec>::from(temp1);
    double c_dT = 0.5*size_graph*log(M_PI) + (0.5*p*b+size_graph)*log(2) + sum(arma::lgamma(0.5*(b+nu2))) + sum( (b+nu2+temp2) % log(Ti.diag()) );
    logIg = c_dT + log_Ef_T;
  }
  
  return logIg;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/* may have to remove these lines before doing Rcpp
getEdgeMat = function(Adj){
  vAdj = apply(Adj, 2, sum)
  Adj[lower.tri(Adj, diag = FALSE)] = 0
  Edge = which(Adj==1, arr.ind = TRUE)
  
  l = c(rep(2, nrow(Edge)), rep(1, length(which(vAdj==0))))
  ind1 = c(Edge[,1], which(vAdj==0)) 
  ind2 = c(Edge[,2], rep(0,length(which(vAdj==0))))
  EdgeMat = cbind(l, ind1, ind2)
  colnames(EdgeMat) = NULL
  return(EdgeMat)
}

gnormJT(G_9, getEdgeMat(G_9), 5, 3*diag(9))
*/

// [[Rcpp::export]]
double log_multi_gamma(int p, double n){
  double f;
  f = 0.25*p*(p-1)*log(M_PI);
  
  for(unsigned int i=1; i<=p; i++){
    f += lgamma(n + 0.5 - 0.5*i); // pay attention to type conversion
  }
  
  return f;
}

// [[Rcpp::export]]
double log_wishart_norm(int p, double b, arma::mat D){
  return 0.5*(b+p-1)*p*log(2.0) - 0.5*(b+p-1)*log(det(D)) + log_multi_gamma(p, 0.5*(b+p-1));
}

/* get the normalizing constant of a G-Wishart density with junction tree */
// Adj's diagonal entries are zero
// [[Rcpp::export]]
double gnormJT(arma::umat Adj, arma::umat EdgeMat, double b, arma::mat D, int iter = 500){
  
  double lC = 0;
  double lS = 0;
  
  vector<arma::uvec> Clist;
  vector<arma::uvec> Slist;
  int nC, nS;
  getJT_C(EdgeMat, Clist, Slist, nC, nS);
  
  for(unsigned int iC=0; iC<nC; iC++){
    if(arma::accu(Adj(Clist[iC]-1, Clist[iC]-1)) == Clist[iC].n_elem*(Clist[iC].n_elem-1)){
      if(Clist[iC].n_elem==1){ 
        // cout<<"1. clique with only one node"<<endl;
        int singleC = Clist[iC](0);
        lC = lC + lgamma(0.5*b) - 0.5*b*log(0.5*D(singleC-1,singleC-1));
      }
      else{ 
        // cout<<"2. clique with more than one node"<<endl;
        lC = lC + log_wishart_norm(Clist[iC].n_elem, b, D(Clist[iC]-1, Clist[iC]-1)); }
    }
    else{ 
      // cout<<"3. non-complete prime component"<<endl;
      lC = lC + gnorm_c(Adj(Clist[iC]-1, Clist[iC]-1), b, D(Clist[iC]-1, Clist[iC]-1), iter);
    }
  }
  
  if(nS!=0){
    for(unsigned int iS=0; iS<nS; iS++){
      if(Slist[iS].n_elem==1){ 
        // cout<<"4. separator with only one node"<<endl;
        int singleS = Slist[iS](0);
        lS = lS + lgamma(0.5*b) - 0.5*b*log(0.5*D(singleS-1,singleS-1)); }
      else{ 
        // cout<<"5. separator with more than one node"<<endl;
        lS = lS + log_wishart_norm(Slist[iS].n_elem, b, D(Slist[iS]-1, Slist[iS]-1)); }
    }
  }
  // cout<<lC<<endl<<lS<<endl;
  return lC - lS;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

