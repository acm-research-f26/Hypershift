# Draft email to the THINK authors (send from your own account)

Address it to the corresponding author listed on p. 849 of `docs/paper/icdm22-think.pdf`. Their email addresses are not recorded in this repo, so copy them from the PDF.

---

**Subject:** Reproducing THINK (ICDM 2022): epoch selection and Sharpe definition

Dear Dr. Sawhney and co-authors,

We are reproducing "THINK: Temporal Hypergraph Hyperbolic Network" (ICDM 2022) for a student research project, using the RSR NYSE/NASDAQ data. The repository linked in footnote 1 (p. 852) is empty, so we reimplemented the model from the paper. Three details would let us compare against Table II correctly:

1. **Model selection.** How was the reported epoch or checkpoint chosen: by validation (2016) performance, by test (2017) performance, or as a fixed final epoch? How many epochs were trained?
2. **Sharpe ratio.** Sec. IV-B gives SR = E[R_a − R_f] / std[R_a − R_f]. Which k (top-k stocks) and which R_f did you use? Is the reported value annualised (for example × √252)?
3. **NDCG.** Was NDCG@k computed per day and averaged over the test days, or with the STHAN-SR evaluator?

If you can share the code, the hyperparameters (learning rate, weight decay, batch size, window), or the reading of the ⊙ operator in eq. 7 and eq. 14, that would also help.

So far, picking the epoch by test-year score reproduces and exceeds your Table II ordering (THINK Sharpe 2.40 annualised vs 1.64 for TCONV+DHHAN). Picking it on the 2016 validation year makes the advantage disappear. We would like to make sure we are not misreading your protocol.

Thank you for your time.

Best regards,
[Your name], [University / project]
