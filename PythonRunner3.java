import java.io.*;

// [3] test Motion
// docName, action 전달하면 docName으로 해당 모델 찾고 action인지 유사도 검사 후 결과 반환 (true - 동작 수행함, false - 시간 초과(동작 수행x)
public class PythonRunner3 {
    public static void main(String[] args) throws IOException, InterruptedException {
        // parameters : document name
        String docName = "testDoc";
        String action = "first";

        // activate anaconda virtual environment, execute python file
        ProcessBuilder pb = new ProcessBuilder();
        pb.command("/Users/seungtoc/anaconda3/bin/conda", "run", "-n", "motion", "python", "testMotion.py", docName, action);
        Process process = pb.start();

        // get outputs
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }

        //  for confirmation
        int exitCode = process.waitFor();
        if (exitCode == 0) {
            System.out.println("testMotion.py executed successfully.");
        } else {
            System.out.println("testMotion.py execution failed.");
        }
    }
}
