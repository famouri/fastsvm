/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fastkernelsvm;

import java.io.File;
import java.util.Scanner;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_problem;
import svm.svm_train;


/**
 *
 * @author Mahmoud J00N
 */
public class FastKernelSVM {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try{
//            svm_problem problem = new svm_problem();
//            svm_node node = new svm_node();
//            node.
            svm_train t = new svm_train();
            t.setParameters("iris.scale", "model.model");
            t.read_problem();
            svm_model model = svm.svm_train(t.prob,t.param);
            svm.svm_save_model("model.model",model);
//            t.do_cross_validation();


        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
    
}
