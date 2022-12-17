
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;


class TransE {

	public boolean L1_flag = true; //false; set false using L2
	public int n = 50; // dimension of vectors
	public int method = 0; //0 means uniform and 1 means bernoulli
	public double rate = 0.01; // learning rate
	public double margin = 1;

    public int relation_num;
    public int entity_num;
    public  Map<String, Integer> relation2id, entity2id;
    public Map<Integer, String> id2entity, id2relation;
    public Map<Integer, Map<Integer, Integer>> left_entity, right_entity; //is an hash map using relation_id as key, the value is the entity_id and the number of the entity_id
    public Map<Integer, Double> left_num, right_num;  //relation_id as key, the value is the (sum of the value of left_entity)/how many head entities
    public Set<Triple<Integer, Integer, Integer>> ok; //an hash set stores all triples

    public double res; //loss function value
    public List<Integer> fb_h, fb_l, fb_r;

    public double[][] relation_vec, entity_vec;
    public double[][] relation_tmp, entity_tmp;

    public void prepare() throws IOException{
    		ok = new HashSet<>();

    		// read the entity2id file
        entity2id = new HashMap<>();
        id2entity = new HashMap<>();
    		int count = 0;
    		File f_e = new File("data/entity2id.txt");
    		BufferedReader reader_e = new BufferedReader( new InputStreamReader (new FileInputStream(f_e), "UTF-8"));
    		String line;
    		while ((line = reader_e.readLine()) != null) {
    			 String[] split_data = line.split("\\s+");
    			 entity2id.put(split_data[0], Integer.valueOf(split_data[1]));
    			 id2entity.put(Integer.valueOf(split_data[1]), split_data[0]);
    			 count++;
    		}
    		entity_num = count;
    		reader_e.close();
    		// read the relation2id file
    		relation2id = new HashMap<>();
        id2relation = new HashMap<>();
        count = 0;
        File f_r = new File("data/relation2id.txt");
		BufferedReader reader_r = new BufferedReader( new InputStreamReader (new FileInputStream(f_r), "UTF-8"));
		String line_r;
		while ((line_r = reader_r.readLine()) != null) {
			 String[] split_data = line_r.split("\\s+");
			 relation2id.put(split_data[0], Integer.valueOf(split_data[1]));
			 id2relation.put(Integer.valueOf(split_data[1]), split_data[0]);
			 count++;
		}
		relation_num = count;
		reader_r.close();

        // read the train file
				left_entity = new HashMap<>();
        right_entity = new HashMap<>();
        fb_h = new ArrayList<>(); // stores all head entity's ids of training data
        fb_r = new ArrayList<>(); // stores all relation's ids of training data
        fb_l = new ArrayList<>(); // stores all tail entity's ids of training data
        File f_t = new File("data/train.txt");
        BufferedReader reader_t = new BufferedReader(new InputStreamReader(new FileInputStream(f_t), "UTF-8"));
        String line_t;
        while ((line_t = reader_t.readLine()) != null) {
        	 	String[] split_data = line_t.split("\\s+");
        	 	String head = split_data[0];
        	 	String tail = split_data[1];
        	 	String relation = split_data[2];
        	 	if (!entity2id.containsKey(head)) {
                     System.out.printf("miss entity: %s\n", head);
                     continue;
        	 	}
            if (!entity2id.containsKey(tail)) {
                   	System.out.printf("miss entity: %s\n", tail);
                     continue;
            }
            if (!relation2id.containsKey(relation)) {
                     relation2id.put(relation, relation_num);
                     relation_num++;
            }
						//create new HashMap if the key relation not exist
            if (!left_entity.containsKey(relation2id.get(relation))) {
                left_entity.put(relation2id.get(relation), new HashMap<>());
            }
						// put a value pair <head: 0> for the <relation: <head: >> if head not exist
            if (!left_entity.get(relation2id.get(relation)).containsKey(entity2id.get(head)))
                    left_entity.get(relation2id.get(relation)).put(entity2id.get(head), 0);
						// <relation: <head: i+1>> add 1 for exist relation and head
						left_entity.get(relation2id.get(relation)).put(entity2id.get(head),
                    left_entity.get(relation2id.get(relation)).get(entity2id.get(head)) + 1);
            //create new HashMap if the key relation not exist
            if (!right_entity.containsKey(relation2id.get(relation)))
                right_entity.put(relation2id.get(relation), new HashMap<>());
						// put a value pair <tail: 0> for the <relation: <tail: >> if tail not exist
            if (!right_entity.get(relation2id.get(relation)).containsKey(entity2id.get(tail)))
                right_entity.get(relation2id.get(relation)).put(entity2id.get(tail), 0);
						// <relation: <tail: i+1>> add 1 for exist relation and tail
            right_entity.get(relation2id.get(relation)).put(entity2id.get(tail),
                    right_entity.get(relation2id.get(relation)).get(entity2id.get(tail)) + 1);
            // call add function
            add(entity2id.get(head), entity2id.get(tail), relation2id.get(relation));
        }
        reader_t.close();
        left_num = new HashMap<>(); //using relation_id as key, the value is the average of head entity number (match the relation in the trainning data)
        right_num = new HashMap<>(); //using relation_id as key, the value is the average of head entity number
        for (int i = 0; i < relation_num; i++) {
        		double sum1 = 0, sum2 = 0;
            for (Entry<Integer, Integer> it : left_entity.get(i).entrySet()) {
            		sum1++;
            		sum2 += it.getValue();
            }
            left_num.put(i, sum2 / sum1);
        }
        for (int i = 0; i < relation_num; i++) {
        		double sum1 = 0, sum2 = 0;
        		for (Entry<Integer, Integer> it : right_entity.get(i).entrySet()) {
        			sum1++;
        			sum2 += it.getValue();
        		}
        		right_num.put(i, sum2 / sum1);
        }

        System.out.println("relation_num = " + relation_num);
        System.out.println("entity_num = " + entity_num);
        System.out.println("Prepare done, now it is time to train");

    }
    
	/*
	** Input : e1: head,  e2: tail, r: relation
	** Output: None
	*/
	private void add(int e1, int e2, int r) {
        fb_h.add(e1);
        fb_r.add(r);
        fb_l.add(e2);
		//create a triple <head, relation, tail>
        Triple<Integer, Integer, Integer> triple = new ImmutableTriple<>(e1, r, e2);
		// add triple into HashSet
		ok.add(triple);
    }

    public void run() throws IOException{
        relation_vec = new double[relation_num][n];
        entity_vec = new double[entity_num][n];
        //initialize the vectors
        for (int i = 0; i < relation_num; i++) {
            for (int ii = 0; ii < n; ii++)
                relation_vec[i][ii] = randn(0, 1.0 / n, -6.0 / Math.sqrt(n), 6.0 / Math.sqrt(n));
            //norm(relation_vec[i]);
        }
        for (int i = 0; i < entity_num; i++) {
            for (int ii = 0; ii < n; ii++)
                entity_vec[i][ii] = randn(0, 1.0 / n, -6.0 / Math.sqrt(n), 6.0 / Math.sqrt(n));
            norm(entity_vec[i]);
        }
        
        bfgs();
    }

    private void bfgs() throws IOException{
        System.out.println("BFGS:");
        //System.out.println(fb_h.size());
        res = 0; // loss function value
        int nbatches = 100; // split training data into nbatches
        int nepoch = 10; // 1000; usually 1000 or something large to converge
        int batchsize = fb_h.size() / nbatches;
        for (int epoch = 0; epoch < nepoch; epoch++) {
            res = 0;
            for (int batch = 0; batch < nbatches; batch++) {

            		//relation_tmp = relation_vec.clone();
            		//entity_tmp = entity_vec.clone();
            		//change shallow copy to deep copy
            		relation_tmp = new double[relation_num][n];
            		entity_tmp = new double[entity_num][n];
            		for (int i = 0; i < relation_num; i++) {
                        for (int ii = 0; ii < n; ii++)
                            relation_tmp[i][ii] = relation_vec[i][ii];
                    }
                for (int i = 0; i < entity_num; i++) {
                        for (int ii = 0; ii < n; ii++)
                            entity_tmp[i][ii] = entity_vec[i][ii];
                    }

                for (int k = 0; k < batchsize; k++) {
                    int i = rand_max(fb_h.size()); // random choose a trainning instance
                    int j = rand_max(entity_num); // negative sample
                    double pr = 1000.0 * right_num.get(fb_r.get(i))
                            / (right_num.get(fb_r.get(i)) + left_num.get(fb_r.get(i)));
                    if (method == 0)
                        pr = 500;
                    if (rand_max(1000) < pr) { // put negative sample j as tail for SGD
												// continue change j until we haven't saw j in ok HashSet
                        while (ok.contains(new ImmutableTriple<>(fb_h.get(i), fb_r.get(i), j)))
                            j = rand_max(entity_num);
												// call train_kb funciton using j as tail
												train_kb(fb_h.get(i), fb_l.get(i), fb_r.get(i), fb_h.get(i), j, fb_r.get(i));
                    } else { // put negative sample j as head for SGD
                        while (ok.contains(new ImmutableTriple<>(j, fb_r.get(i), fb_l.get(i))))
                            j = rand_max(entity_num);
                        train_kb(fb_h.get(i), fb_l.get(i), fb_r.get(i), j, fb_l.get(i), fb_r.get(i));
                    }
                    norm(relation_tmp[fb_r.get(i)]);
                    norm(entity_tmp[fb_h.get(i)]);
                    norm(entity_tmp[fb_l.get(i)]);
                    norm(entity_tmp[j]);
                }
                //relation_vec = relation_tmp;
                //entity_vec = entity_tmp;
                // deep copy back
                for (int i = 0; i < relation_num; i++) {
                    for (int ii = 0; ii < n; ii++)
                        relation_vec[i][ii] = relation_tmp[i][ii];
                }
                for (int i = 0; i < entity_num; i++) {
                    for (int ii = 0; ii < n; ii++)
                        entity_vec[i][ii] = entity_tmp[i][ii];
                }

            }
            System.out.println("epoch:" + epoch + ' ' + res);
        }
        //write vectors to the file
        	String version = new String();
        if (method == 0) {
        		version = "unif";
        } else {
        		version = "bern";
        }
        Write_Vec2File("data/relation2vec."+ version, relation_vec, relation_num);
        Write_Vec2File("data/entity2vec."+version, entity_vec, entity_num);

    }
	/*** Input: e1: head, e2: tail, rel: relation
	**** calculate: tail - head - relation
	*** Output: None
	***/
    private double calc_sum(int e1, int e2, int rel) {
        double sum = 0;
        if (L1_flag) {
            for (int ii = 0; ii < n; ii++) { // n is the dimension of vector
                sum += Math.abs(entity_vec[e2][ii] - entity_vec[e1][ii] - relation_vec[rel][ii]);
            }
        }else {
            for (int ii = 0; ii < n; ii++) {
                sum += Math.pow(entity_vec[e2][ii] - entity_vec[e1][ii] - relation_vec[rel][ii], 2);
            }
        }
        return sum;
    }
	/*** Input:
	**** e1_a: head, e2_a: tail, rel_a: relation <e1_a, e2_a, rel_a> is an sample from training data
	**** e1_b: head, e2_b: tail, rel_b: relation <e1_b, e2_b, rel_b> is an negative sample
	*** update vectors accordingly 
	*** Output: None
	***/
    private void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
        for (int ii = 0; ii < n; ii++) {
            double x = 2.0 * (entity_vec[e2_a][ii] - entity_vec[e1_a][ii] - relation_vec[rel_a][ii]);
            if (L1_flag) {
                if (x > 0) {
                    x = 1;
                }else {
                    x = -1;}
            }
            relation_tmp[rel_a][ii] -= -1.0 * rate * x;
            entity_tmp[e1_a][ii] -= -1.0 * rate * x;
            entity_tmp[e2_a][ii] += -1.0 * rate * x;
            x = 2.0 * (entity_vec[e2_b][ii] - entity_vec[e1_b][ii] - relation_vec[rel_b][ii]);
            if (L1_flag) {
                if (x > 0) {
                    x = 1;
                }else {
                    x = -1;}
            }
            relation_tmp[rel_b][ii] -= rate * x;
            entity_tmp[e1_b][ii] -= rate * x;
            entity_tmp[e2_b][ii] += rate * x;
        }
    }
	/*** Input:
	**** e1_a: head, e2_a: tail, rel_a: relation <e1_a, e2_a, rel_a> is an sample from trainning data
	**** e1_b: head, e2_b: tail, rel_b: relation <e1_b, e2_b, rel_b> is an negative sample
	*** Output: None
	***/
    private void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
        double sum1 = calc_sum(e1_a, e2_a, rel_a);
        double sum2 = calc_sum(e1_b, e2_b, rel_b);
        if (sum1 + margin > sum2) {
            res += margin + sum1 - sum2;
			// call gradient function
            gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
        }
    }

    private static double rand(double min, double max) {
        return min + (max - min) * Math.random();
    }

    private static double normal(double x, double miu, double sigma) {
        return 1.0 / Math.sqrt(2.0 * Math.PI) / sigma * Math.exp(-1.0 * (x - miu) * (x - miu) / (2.0 * sigma * sigma));
    }

    private static double randn(double miu, double sigma, double min, double max) {
        double x, y, dScope;
        do {
            x = rand(min, max);
            y = normal(x, miu, sigma);
            dScope = rand(0.0, normal(miu, miu, sigma));
        } while (dScope > y);
        return x;
    }

    private static double vec_len(double[] a) {
        double res = 0;
        for (double anA : a) res += anA * anA;
        res = Math.sqrt(res);
        return res;
    }

    private static void norm(double[] a) {
        double x = vec_len(a);
        if (x > 1) {
            for (int ii = 0; ii < a.length; ii++) {
                a[ii] /= x;
            }
        }
    }

    private static int rand_max(int x) {
        return (int) (Math.random() * x);
    }
	/*** Input:
	**** file_name
	**** vec: entity_vec or relation_vec
	**** number: entity_num or relation_vec 
	*** Output: None
	***/
    private void Write_Vec2File(String file_name, double[][] vec, int number) throws IOException {
        File f = new File(file_name);
        OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(f), "UTF-8");
        for (int i = 0; i < number; i++) {
            for (int j = 0; j < n; j++) {
                String str = String.format("%.6f\t", vec[i][j]);
                writer.write(str);
            }
            writer.write("\n");
        }
        		writer.flush();
            writer.close();
    }

} // end of TransE class

class Main {

public static void main(String[] args) throws IOException{
		System.out.println("Begin TransE trainning");
		long startTime = System.nanoTime();
		TransE transe = new TransE();
		transe.prepare();
		transe.run();
		System.out.println("All done!!");
		long endTime   = System.nanoTime();
		long totalTime = endTime - startTime;
		double seconds = (double)totalTime / 1_000_000_000.0;
		System.out.println(seconds);

	}
}