/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package java8;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 *
 * @author Reggie Barker
 */
public class Java8 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {
        // TODO code application logic here
        List <String> listaA = new ArrayList<>(abrirArchivo());
        List<String> listaB = new ArrayList<>(listaA);
        int res = listaA.parallelStream().map(i->conPalabra(i, listaB)).reduce(Integer::sum).get();
        System.out.println("Palabras que contiene otras palabras: " + res);
        int res2 = listaA.parallelStream().map(i->conSubPalabra(i, listaB)).reduce(Integer::sum).get();
        System.out.println("Palabras que contiene otras palabras: " + res2);
        int res3 = listaA.parallelStream().map(i->conVocales(i, listaB)).reduce(Integer::sum).get();
        System.out.println("Palabras que contiene otras palabras: " + res3);
    }
    public static int conVocales(String a, List<String> b){
        List <String> listaA = new ArrayList<>();
        for (int i = 0; i < a.length(); i++) {
            if((a.charAt(i)== 'a')  || (a.charAt(i)== 'e') || (a.charAt(i)== 'i') || (a.charAt(i)== 'o') || (a.charAt(i)== 'u')){
                String gg =  Character.toString(a.charAt(i));
                List<String> lista = conSubPalabraAux(gg, b);
                lista.forEach((g) -> {
                listaA.add(g);
            });
            }
            
        }
        return 0;
    }
    
    public static int conSubPalabra(String a, List<String> b){
        String subString = "";
        List <String> listaA = new ArrayList<>();
        for (int j = 0; j < a.length(); j++) {
            subString += a.charAt(j);
            List<String> lista = conSubPalabraAux(subString, b);
            lista.forEach((g) -> {
                listaA.add(g);
            });
            
        }
        return listaA.size()-1;
    }
public static List<String> conSubPalabraAux(String a, List<String> b){
    List <String> lista = b.parallelStream().filter(i->i.contains(a)).collect(Collectors.toList());
    return lista;
    
}
    public static int conPalabra(String a, List<String> b){
        List<String> lista = b.parallelStream().filter(i->i.contains(a)).collect(Collectors.toList());
        return lista.size()-1;
    }
    
    public static List<String> abrirArchivo() throws FileNotFoundException, IOException{
        File archivo = null;
        FileReader fr = null;
        BufferedReader br = null;
        List<String> lista = new ArrayList <>();
        archivo = new File ("words.txt");
        fr = new FileReader (archivo);
        br = new BufferedReader(fr);
        String linea;
        while((linea=br.readLine())!=null){
            lista.add(linea);
        }
        
        return lista;
    }
    
}
