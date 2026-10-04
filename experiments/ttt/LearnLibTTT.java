import java.io.*;
import java.util.*;
import de.learnlib.algorithm.ttt.mealy.TTTLearnerMealy;
import de.learnlib.acex.AcexAnalyzers;
import de.learnlib.oracle.MembershipOracle;
import de.learnlib.query.DefaultQuery;
import de.learnlib.query.Query;
import net.automatalib.alphabet.impl.Alphabets;
import net.automatalib.automaton.transducer.MealyMachine;
import net.automatalib.word.Word;

/** Actual LearnLib TTT. No simulator/reference transition table enters this process. */
public final class LearnLibTTT {
    static final BufferedReader IN = new BufferedReader(new InputStreamReader(System.in));
    static String read() {
        try { String line = IN.readLine(); if (line == null) throw new IllegalStateException("oracle closed"); return line; }
        catch (IOException e) { throw new UncheckedIOException(e); }
    }
    static Word<Integer> parseWord(String line) {
        if (line.isEmpty()) return Word.epsilon();
        List<Integer> values = new ArrayList<>();
        for (String s : line.split(" ")) values.add(Integer.valueOf(s));
        return Word.fromList(values);
    }
    static void send(String s) { System.out.println(s); System.out.flush(); }
    static final class Oracle implements MembershipOracle<Integer, Word<Integer>> {
        public void processQueries(Collection<? extends Query<Integer, Word<Integer>>> queries) {
            for (Query<Integer, Word<Integer>> q : queries) {
                StringBuilder msg = new StringBuilder("MQ ").append(q.getPrefix().length());
                for (Integer i : q.getInput()) msg.append(' ').append(i);
                send(msg.toString());
                Word<Integer> out = parseWord(read());
                if (out.length() != q.getSuffix().length()) throw new IllegalStateException("wrong MQ output length");
                q.answer(out);
            }
        }
    }
    static <S,T> String export(MealyMachine<S,Integer,T,Integer> h, int alphabetSize) {
        List<S> states = new ArrayList<>(h.getStates());
        Map<S,Integer> ids = new HashMap<>();
        for (int i = 0; i < states.size(); i++) ids.put(states.get(i), i);
        StringBuilder b = new StringBuilder("{\"initial\":").append(ids.get(h.getInitialState())).append(",\"transitions\":[");
        for (int i = 0; i < states.size(); i++) {
            if (i > 0) b.append(','); b.append('[');
            for (int a = 0; a < alphabetSize; a++) {
                if (a > 0) b.append(',');
                T t = h.getTransition(states.get(i), a);
                if (t == null) throw new IllegalStateException("partial hypothesis");
                b.append('[').append(ids.get(h.getSuccessor(t))).append(',').append(h.getTransitionOutput(t)).append(']');
            }
            b.append(']');
        }
        return b.append("]}").toString();
    }
    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        TTTLearnerMealy<Integer,Integer> learner = new TTTLearnerMealy<>(Alphabets.integers(0,n-1), new Oracle(), AcexAnalyzers.BINARY_SEARCH_BWD);
        learner.startLearning();
        int refinements = 0;
        for (;;) {
            String model = export(learner.getHypothesisModel(), n);
            send("EQ " + model);
            String response = read();
            if (response.equals("OK")) {
                send("DONE {\"learner_class\":\"" + learner.getClass().getName() + "\",\"refinements\":" + refinements + ",\"hypothesis\":" + model + "}");
                return;
            }
            if (!response.startsWith("CE ")) throw new IllegalStateException("bad EQ reply");
            Word<Integer> input = parseWord(response.substring(3));
            Word<Integer> output = parseWord(read());
            if (output.length() != input.length()) throw new IllegalStateException("wrong CE output length");
            if (!learner.refineHypothesis(new DefaultQuery<Integer,Word<Integer>>(Word.epsilon(), input, output))) throw new IllegalStateException("counterexample did not refine");
            refinements++;
        }
    }
}
