#ifndef SIMPLEX_H
#define SIMPLEX_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>

class noMinimizer: public std::exception{
  virtual const char* what() const throw(){
    return "No minimizer in simplex";
  }
};

bool greater(double a, double b){ //T if a>b
    return (a - b) > 1e-10;
}
bool equals(double a, double b){ //T if a==b
    return std::abs(a - b) <= 1e-10;
}
bool less(double a, double b){ //T if a<b
    return (b - a) > 1e-10;
}

void simplexSolve(Eigen::MatrixXd& tableau, int s){
    std::vector<int> identity(tableau.rows());
    identity[0]=0;
    for(int j = s;j<tableau.cols();j++){
        bool b = false;
        int idx = -1;
        for(int i = s;i<tableau.rows();i++){
            if(equals(tableau(i,j),0))continue;
            else if(equals(tableau(i,j),1)&&idx==-1){
                idx = i;
            }else{
                b = true;
                break;
            }
        }
        if(b)continue;
        identity[idx]=j;
    }
    for(int i = s;i<identity.size();i++){//pricing out
        tableau.row(0)-=tableau(0,identity[i])*tableau.row(i);
    }
    while(true){
        int idx1 = -1; //pivot column
        int idx2 = -1; //pivot row
        for(int i = s;i<tableau.cols()-1;i++){
            if((idx1==-1&&greater(tableau(0,i),0))||(idx1!=-1&&greater(tableau(0,i),tableau(0,idx1)))){
                idx1 = i;
            }
        }
        if(idx1==-1)break;
        for(int i = s;i<tableau.rows();i++){
            if((idx2==-1&&greater(tableau(i,idx1),0))||(idx2!=-1&&greater(tableau(i,idx1),0)&&less(tableau(i,tableau.cols()-1)/tableau(i,idx1),tableau(idx2,tableau.cols()-1)/tableau(idx2,idx1)))){
                idx2 = i;
            }
        }
        if(idx2==-1){
            throw noMinimizer();
        }
        tableau.row(idx2)/=tableau(idx2,idx1);
        for(int i = 0;i<tableau.rows();i++){
            if(i!=idx2){
                tableau.row(i)-=tableau(i,idx1)*tableau.row(idx2);
            }
        }
    }
}

Eigen::VectorXd simplex(Eigen::VectorXd cost, Eigen::MatrixXd constraints, std::vector<char> inequalities){
    int s = 0, a = 0;
    for(int i = 0;i<constraints.rows();i++){
        if(constraints(i,constraints.cols()-1)<0){
            constraints.row(i)*=-1;
            if(inequalities[i]=='<'){
                inequalities[i]='>';
                s++;
                a++;
            }else if(inequalities[i]=='>'){
                inequalities[i]='<';
                s++;
            }else{
                a++;
            }
        }else{
            if(inequalities[i]=='<'){
                s++;
            }else if(inequalities[i]=='>'){
                a++;
                s++;
            }else{
                a++;
            }
        }
    }
    Eigen::MatrixXd slack = Eigen::MatrixXd::Zero(constraints.rows(),s);
    Eigen::MatrixXd artificial = Eigen::MatrixXd::Zero(constraints.rows(),a);
    s = 0;
    a = 0;
    for(int i = 0;i<inequalities.size();i++){
        if(inequalities[i]=='<'){
            slack(i,s) = 1;
            s++;
        }else if(inequalities[i]=='>'){
            slack(i,s) = -1;
            artificial(i,a) = 1;
            s++;
            a++;
        }else{
            artificial(i,a) = 1;
            a++;
        }
    }
    Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(constraints.rows()+2,constraints.cols()+slack.cols()+artificial.cols()+2);
    temp(0,0) = 1;
    temp(1,1) = 1;
    temp.block(1,2,1,cost.size())=-cost.transpose();
    temp.block(2,2,constraints.rows(),constraints.cols()-1)=constraints.block(0,0,constraints.rows(),constraints.cols()-1);
    temp.block(2,1+constraints.cols(),slack.rows(),slack.cols())=slack;
    temp.block(2,1+constraints.cols()+slack.cols(),artificial.rows(),artificial.cols())=artificial;
    temp.block(0,1+constraints.cols()+slack.cols(),1,artificial.cols())=-Eigen::MatrixXd::Ones(1,artificial.cols());
    temp.block(2,temp.cols()-1,constraints.rows(),1)=constraints.col(constraints.cols()-1);
    Eigen::MatrixXd tableau;
    //PHASE 1
    if(a>0){
        simplexSolve(temp,2);
        tableau = Eigen::MatrixXd(constraints.rows()+1,constraints.cols()+slack.cols()+1);
        tableau << temp.block(1,1,constraints.rows()+1,constraints.cols()+slack.cols()), temp.block(1,temp.cols()-1,constraints.rows()+1,1);
    }else{
        tableau = temp.block(1,1,temp.rows()-1,temp.cols()-1);
    }
    //PHASE 2
    simplexSolve(tableau,1);
    Eigen::VectorXd ans = Eigen::VectorXd::Zero(cost.size());
    for(int j = 1;j<=cost.size();j++){
        int idx = -1;
        for(int i = 1;i<tableau.rows();i++){
            if(tableau(i,j)==1&&idx==-1){
                idx = i;
            }else if(tableau(i,j)!=0){
                idx = -1;
                break;
            }
        }
        if(idx!=-1)ans[j-1] = tableau(idx,tableau.cols()-1);
    }
    return ans;
}

#endif //SIMPLEX_H
