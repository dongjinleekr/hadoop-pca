package com.github.dongjinleekr.hadoop.covariance;

import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.spectral.common.IntDoublePairWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Stopwatch;

public class CovarianceJob extends AbstractJob {
  private static final Logger logger = LoggerFactory.getLogger(CovarianceJob.class); 

  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    addInputOption();
    addOutputOption();
    addOption(CovarianceOptionCreator.cardinalityOption().create());
    addOption(CovarianceOptionCreator.vectorCountOption().create());
    
    if (parseArguments(args) == null) {
      return -1;
    }
    
    if (null == getConf()) {
      setConf(new Configuration());
    }
    
    Path input = getInputPath();
    Path output = getOutputPath();
    int cardinality = Integer.parseInt(getOption(CovarianceOptionCreator.CARDINALITY));
    int vectorCount = Integer.parseInt(getOption(CovarianceOptionCreator.VECTOR_COUNT));
     
    return run(getConf(), input, output, cardinality, vectorCount);
  }
  
  public int run(Configuration conf, Path input, Path output, int cardinality, int vectorCount) throws IOException, ClassNotFoundException, InterruptedException {
    conf.setInt(CovarianceOptionCreator.CARDINALITY, cardinality);
    conf.setInt(CovarianceOptionCreator.VECTOR_COUNT, vectorCount);
    
    // current datetime
    DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
    Date current = new Date();

    // task name
    String jobName = String.format("covariance (%s)", dateFormat.format(current));
    logger.info(String.format("Job Name: %s", jobName));

    // thread count
    Runtime runtime = Runtime.getRuntime();
    int processorCount = runtime.availableProcessors();
    logger.info(String.format("processorCount count: %s", processorCount));
    int threadCount = 2 * processorCount;
    logger.info(String.format("thread count: %s", processorCount));

    // create Job
    Job job = new Job(conf);
    
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(IntDoublePairWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    job.setMapperClass(MultithreadedMapper.class);
    MultithreadedMapper.setMapperClass(job, Covariance.CovMapper.class);
    MultithreadedMapper.setNumberOfThreads(job, threadCount);
    job.setReducerClass(Covariance.CovReducer.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);
    
    job.setJarByClass(getClass());
    job.setJobName(jobName);

    return job.waitForCompletion(true) ? 0 : 1;
  }

  public static int main(String[] args) throws Exception {
    // execute Job
    Stopwatch stopwatch = new Stopwatch().start();
    int ret = ToolRunner.run(new CovarianceJob(), args);
    stopwatch.stop();

    logger.info(String.format("Time Elapsed: %d (ms)", stopwatch.elapsedMillis()));

    return ret;
  }
}
